from functools import reduce
import unittest
import testscenarios
from os_ken.ofproto import ofproto_v1_5
from os_ken.ofproto import ofproto_v1_5_parser
def _list_test_cases():
    import functools
    import itertools

    class Field(object):
        pass

    class Int1(Field):

        @staticmethod
        def generate():
            yield 0
            yield 255

    class Int2(Field):

        @staticmethod
        def generate():
            yield 0
            yield 4660
            yield 65535

    class Int3(Field):

        @staticmethod
        def generate():
            yield 0
            yield 1193046
            yield 16777215

    class Int4(Field):

        @staticmethod
        def generate():
            yield 0
            yield 305419896
            yield 4294967295

    class Int4double(Field):

        @staticmethod
        def generate():
            yield [0, 1]
            yield [305419896, 591751049]
            yield [4294967295, 4294967294]

    class Int8(Field):

        @staticmethod
        def generate():
            yield 0
            yield 1311768467463790320
            yield 18446744073709551615

    class Mac(Field):

        @staticmethod
        def generate():
            yield '00:00:00:00:00:00'
            yield 'f2:0b:a4:7d:f8:ea'
            yield 'ff:ff:ff:ff:ff:ff'

    class IPv4(Field):

        @staticmethod
        def generate():
            yield '0.0.0.0'
            yield '192.0.2.1'
            yield '255.255.255.255'

    class IPv6(Field):

        @staticmethod
        def generate():
            yield '::'
            yield 'fe80::f00b:a4ff:fed0:3f70'
            yield 'ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff'

    class B64(Field):

        @staticmethod
        def generate():
            yield 'aG9nZWhvZ2U='
            yield 'ZnVnYWZ1Z2E='
    ofpps = [ofproto_v1_5_parser]
    common = [('duration', Int4double), ('idle_time', Int4double), ('flow_count', Int4), ('packet_count', Int8), ('byte_count', Int8), ('field_100', B64)]
    L = {}
    L[ofproto_v1_5_parser] = common

    def flatten_one(l, i):
        if isinstance(i, tuple):
            return l + flatten(i)
        else:
            return l + [i]
    flatten = lambda l: reduce(flatten_one, l, [])
    cases = []
    for ofpp in ofpps:
        for n in range(1, 3):
            for C in itertools.combinations(L[ofpp], n):
                l = [1]
                keys = []
                clss = []
                for k, cls in C:
                    l = itertools.product(l, cls.generate())
                    keys.append(k)
                    clss.append(cls)
                l = [flatten(x)[1:] for x in l]
                for values in l:
                    d = dict(zip(keys, values))
                    for n, uv in d.items():
                        if isinstance(uv, list):
                            d[n] = tuple(uv)
                    mod = ofpp.__name__.split('.')[-1]
                    method_name = 'test_' + mod
                    for k in sorted(dict(d).keys()):
                        method_name += '_' + str(k)
                        method_name += '_' + str(d[k])
                    method_name = method_name.replace(':', '_')
                    method_name = method_name.replace('.', '_')
                    method_name = method_name.replace('(', '_')
                    method_name = method_name.replace(')', '_')
                    method_name = method_name.replace(',', '_')
                    method_name = method_name.replace("'", '_')
                    method_name = method_name.replace(' ', '_')
                    cases.append({'name': method_name, 'ofpp': ofpp, 'd': d})
    return cases