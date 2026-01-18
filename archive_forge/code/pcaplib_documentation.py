import struct
import sys
import time

    PCAP file writer

    ========== ==================================================
    Argument   Description
    ========== ==================================================
    file_obj   File object which writing PCAP file in binary mode
    snaplen    Max length of captured packets (in octets)
    network    Data link type. (e.g. 1 for Ethernet,
               see `tcpdump.org`_ for details)
    ========== ==================================================

    .. _tcpdump.org: http://www.tcpdump.org/linktypes.html

    Example of usage::

        ...
        from os_ken.lib import pcaplib


        class SimpleSwitch13(app_manager.OSKenApp):
            OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

            def __init__(self, *args, **kwargs):
                super(SimpleSwitch13, self).__init__(*args, **kwargs)
                self.mac_to_port = {}

                # Create pcaplib.Writer instance with a file object
                # for the PCAP file
                self.pcap_writer = pcaplib.Writer(open('mypcap.pcap', 'wb'))

            ...

            @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
            def _packet_in_handler(self, ev):
                # Dump the packet data into PCAP file
                self.pcap_writer.write_pkt(ev.msg.data)

                ...
    