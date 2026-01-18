import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
 Test case for Request-Reply messages.

        Some tests need attached port to switch.
        If use the OVS, can do it with the following commands.
            # ip link add <port> type dummy
            # ovs-vsctl add-port <bridge> <port>
    