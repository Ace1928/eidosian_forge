import errno
import logging
import json
import threading
import time
from queue import PriorityQueue, Empty
import websocket
from parlai.mturk.core.shared_utils import print_and_log
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
class StaticSocketManager(SocketManager):
    """
    Version of SocketManager that communicates consistently with the world, but isn't
    keeping track of the liveliness of the agents that connect as these are single
    person tasks.

    Submissions are handled via post rather than served over socket, so it doesn't make
    sense to.
    """

    def channel_thread(self):
        """
        Handler thread for monitoring all channels to send things to.
        """
        while not self.is_shutdown:
            for connection_id in self.run.copy():
                if not self.run[connection_id]:
                    continue
                try:
                    if connection_id not in self.queues:
                        self.run[connection_id] = False
                        break
                    if self.blocking_packets.get(connection_id) is not None:
                        packet_item = self.blocking_packets[connection_id]
                        if not self.packet_should_block(packet_item):
                            self.blocking_packets[connection_id] = None
                        else:
                            continue
                    try:
                        item = self.queues[connection_id].get(block=False)
                        t = item[0]
                        if time.time() < t:
                            self._safe_put(connection_id, item)
                        else:
                            packet = item[1]
                            if not packet:
                                continue
                            if packet.status is not Packet.STATUS_ACK:
                                self._send_packet(packet, connection_id, t)
                    except Empty:
                        pass
                except Exception as e:
                    shared_utils.print_and_log(logging.WARN, 'Unexpected error occurred in socket handling thread: {}'.format(repr(e)), should_print=True)
            time.sleep(shared_utils.THREAD_SHORT_SLEEP)

    def open_channel(self, worker_id, assignment_id):
        """
        Opens a channel for a worker on a given assignment, doesn't re-open if the
        channel is already open.
        """
        connection_id = '{}_{}'.format(worker_id, assignment_id)
        if connection_id in self.queues and self.run[connection_id]:
            shared_utils.print_and_log(logging.DEBUG, 'Channel ({}) already open'.format(connection_id))
            return
        self.run[connection_id] = True
        self.queues[connection_id] = PriorityQueue()
        self.worker_assign_ids[connection_id] = (worker_id, assignment_id)