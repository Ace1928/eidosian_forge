import logging
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGP_PROCESSOR_ERROR_CODE
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.utils import circlist
from os_ken.services.protocols.bgp.utils.evtlet import EventletIOFactory
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_LOCAL_PREF
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGINATOR_ID
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_CLUSTER_LIST
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_IGP
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_EGP
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_INCOMPLETE
from os_ken.services.protocols.bgp.constants import VRF_TABLE
class BgpProcessor(Activity):
    """Worker that processes queued `Destination'.

    `Destination` that have updates related to its paths need to be
    (re)processed. Only one instance of this processor is enough for normal
    cases. If you want more control on which destinations get processed faster
    compared to other destinations, you can create several instance of this
    works to achieve the desired work flow.
    """
    MAX_DEST_PROCESSED_PER_CYCLE = 100
    _DestQueue = circlist.CircularListType(next_attr_name='next_dest_to_process', prev_attr_name='prev_dest_to_process')

    def __init__(self, core_service, work_units_per_cycle=None):
        Activity.__init__(self)
        self._core_service = core_service
        self._dest_queue = BgpProcessor._DestQueue()
        self._rtdest_queue = BgpProcessor._DestQueue()
        self.dest_que_evt = EventletIOFactory.create_custom_event()
        self.work_units_per_cycle = work_units_per_cycle or BgpProcessor.MAX_DEST_PROCESSED_PER_CYCLE

    def _run(self, *args, **kwargs):
        while True:
            LOG.debug('Starting new processing run...')
            self._process_rtdest()
            self._process_dest()
            if self._dest_queue.is_empty():
                self.dest_que_evt.clear()
                self.dest_que_evt.wait()
            else:
                self.pause(0)

    def _process_dest(self):
        dest_processed = 0
        LOG.debug('Processing destination...')
        while dest_processed < self.work_units_per_cycle and (not self._dest_queue.is_empty()):
            next_dest = self._dest_queue.pop_first()
            if next_dest:
                next_dest.process()
                dest_processed += 1

    def _process_rtdest(self):
        LOG.debug('Processing RT NLRI destination...')
        if self._rtdest_queue.is_empty():
            return
        else:
            processed_any = False
            while not self._rtdest_queue.is_empty():
                next_dest = self._rtdest_queue.pop_first()
                if next_dest:
                    next_dest.process()
                    processed_any = True
            if processed_any:
                self._core_service.update_rtfilters()

    def enqueue(self, destination):
        """Enqueues given destination for processing.

        Given instance should be a valid destination.
        """
        if not destination:
            raise BgpProcessorError('Invalid destination %s.' % destination)
        dest_queue = self._dest_queue
        if destination.route_family == RF_RTC_UC:
            dest_queue = self._rtdest_queue
        if not dest_queue.is_on_list(destination):
            dest_queue.append(destination)
        self.dest_que_evt.set()