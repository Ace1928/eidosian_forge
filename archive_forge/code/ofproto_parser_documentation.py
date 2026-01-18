import base64
import collections
import logging
import struct
import functools
from os_ken import exception
from os_ken import utils
from os_ken.lib import stringify
from os_ken.ofproto import ofproto_common

    This is a base class for OpenFlow message classes.

    An instance of this class has at least the following attributes.

    ========= ==============================
    Attribute Description
    ========= ==============================
    datapath  A os_ken.controller.controller.Datapath instance for this message
    version   OpenFlow protocol version
    msg_type  Type of OpenFlow message
    msg_len   Length of the message
    xid       Transaction id
    buf       Raw data
    ========= ==============================
    