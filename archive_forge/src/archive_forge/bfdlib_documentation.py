import logging
import time
import random
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.ofproto.ether import ETH_TYPE_IP, ETH_TYPE_ARP
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import inet
from os_ken.lib import hub
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import udp
from os_ken.lib.packet import bfd
from os_ken.lib.packet import arp
from os_ken.lib.packet.arp import ARP_REQUEST, ARP_REPLY

        Establish a new BFD session and return My Discriminator of new session.

        Configure the BFD session with the following arguments.

        ================ ======================================================
        Argument         Description
        ================ ======================================================
        dpid             Datapath ID of the BFD interface.
        ofport           Openflow port number of the BFD interface.
        src_mac          Source MAC address of the BFD interface.
        src_ip           Source IPv4 address of the BFD interface.
        dst_mac          (Optional) Destination MAC address of the BFD
                         interface.
        dst_ip           (Optional) Destination IPv4 address of the BFD
                         interface.
        auth_type        (Optional) Authentication type.
        auth_keys        (Optional) A dictionary of authentication key chain
                         which key is an integer of *Auth Key ID* and value
                         is a string of *Password* or *Auth Key*.
        ================ ======================================================

        Example::

            add_bfd_session(dpid=1,
                            ofport=1,
                            src_mac="01:23:45:67:89:AB",
                            src_ip="192.168.1.1",
                            dst_mac="12:34:56:78:9A:BC",
                            dst_ip="192.168.1.2",
                            auth_type=bfd.BFD_AUTH_KEYED_SHA1,
                            auth_keys={1: "secret key 1",
                                       2: "secret key 2"})
        