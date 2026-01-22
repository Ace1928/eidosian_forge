import pickle
import struct
import unittest
from shapely import wkb, wkt
from shapely.geometry import Point
class PersistTestCase(unittest.TestCase):

    def test_pickle(self):
        p = Point(0.0, 0.0)
        data = pickle.dumps(p)
        q = pickle.loads(data)
        assert q.equals(p)

    def test_wkb(self):
        p = Point(0.0, 0.0)
        wkb_big_endian = wkb.dumps(p, big_endian=True)
        wkb_little_endian = wkb.dumps(p, big_endian=False)
        assert p.equals(wkb.loads(wkb_big_endian))
        assert p.equals(wkb.loads(wkb_little_endian))

    def test_wkb_dumps_endianness(self):
        p = Point(0.5, 2.0)
        wkb_big_endian = wkb.dumps(p, big_endian=True)
        wkb_little_endian = wkb.dumps(p, big_endian=False)
        assert wkb_big_endian != wkb_little_endian
        assert wkb_big_endian[0] == 0
        assert wkb_little_endian[0] == 1
        double_size = struct.calcsize('d')
        assert wkb_big_endian[-2 * double_size:] == struct.pack('>2d', p.x, p.y)
        assert wkb_little_endian[-2 * double_size:] == struct.pack('<2d', p.x, p.y)

    def test_wkt(self):
        p = Point(0.0, 0.0)
        text = wkt.dumps(p)
        assert text.startswith('POINT')
        pt = wkt.loads(text)
        assert pt.equals(p)