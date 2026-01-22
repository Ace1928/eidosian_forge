import binascii
import unittest
import struct
from os_ken import exception
from os_ken.ofproto import ofproto_common, ofproto_parser
from os_ken.ofproto import ofproto_v1_0, ofproto_v1_0_parser
import logging
class Datapath(object):
    ofproto = ofproto_v1_0
    ofproto_parser = ofproto_v1_0_parser