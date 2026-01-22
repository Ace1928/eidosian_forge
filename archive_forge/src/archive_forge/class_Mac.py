from functools import reduce
import unittest
import testscenarios
from os_ken.ofproto import ofproto_v1_5
from os_ken.ofproto import ofproto_v1_5_parser
class Mac(Field):

    @staticmethod
    def generate():
        yield '00:00:00:00:00:00'
        yield 'f2:0b:a4:7d:f8:ea'
        yield 'ff:ff:ff:ff:ff:ff'