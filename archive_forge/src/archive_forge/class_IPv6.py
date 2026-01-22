from functools import reduce
import unittest
import testscenarios
from os_ken.ofproto import ofproto_v1_5
from os_ken.ofproto import ofproto_v1_5_parser
class IPv6(Field):

    @staticmethod
    def generate():
        yield '::'
        yield 'fe80::f00b:a4ff:fed0:3f70'
        yield 'ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff'