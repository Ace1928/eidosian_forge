import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
class TestEccPoint_NIST_P256(unittest.TestCase):
    """Tests defined in section 4.3 of https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.204.9073&rep=rep1&type=pdf"""
    pointS = EccPoint(100477533340815411662634551128749658785907042636435106397366501380429417453513, 87104997799923409786648856004022766656120419079854375215656946413621911659094)
    pointT = EccPoint(38744637563132252572193375526521585173096338380822965394069276390274998769771, 38053931953835384495674052639602881660154657110782968445504801383088376660758)

    def test_set(self):
        pointW = EccPoint(0, 0)
        pointW.set(self.pointS)
        self.assertEqual(pointW, self.pointS)

    def test_copy(self):
        pointW = self.pointS.copy()
        self.assertEqual(pointW, self.pointS)
        pointW.set(self.pointT)
        self.assertEqual(pointW, self.pointT)
        self.assertNotEqual(self.pointS, self.pointT)

    def test_negate(self):
        negS = -self.pointS
        sum = self.pointS + negS
        self.assertEqual(sum, self.pointS.point_at_infinity())

    def test_addition(self):
        pointRx = 51876823396606567385074157065323717879096505862684305408641575318717059434110
        pointRy = 63932234198967235656258550897228394755510754316810467717045609836559811134052
        pointR = self.pointS + self.pointT
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)
        pai = pointR.point_at_infinity()
        pointR = self.pointS + pai
        self.assertEqual(pointR, self.pointS)
        pointR = pai + self.pointS
        self.assertEqual(pointR, self.pointS)
        pointR = pai + pai
        self.assertEqual(pointR, pai)

    def test_inplace_addition(self):
        pointRx = 51876823396606567385074157065323717879096505862684305408641575318717059434110
        pointRy = 63932234198967235656258550897228394755510754316810467717045609836559811134052
        pointR = self.pointS.copy()
        pointR += self.pointT
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)
        pai = pointR.point_at_infinity()
        pointR = self.pointS.copy()
        pointR += pai
        self.assertEqual(pointR, self.pointS)
        pointR = pai.copy()
        pointR += self.pointS
        self.assertEqual(pointR, self.pointS)
        pointR = pai.copy()
        pointR += pai
        self.assertEqual(pointR, pai)

    def test_doubling(self):
        pointRx = 53560026360838967581133402626157898931548054975900887465990507413981594986416
        pointRy = 113317629469493007762119134841605581527419933250488044171463979626188406640839
        pointR = self.pointS.copy()
        pointR.double()
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)
        pai = self.pointS.point_at_infinity()
        pointR = pai.copy()
        pointR.double()
        self.assertEqual(pointR, pai)
        pointR = self.pointS.copy()
        pointR += pointR
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)

    def test_scalar_multiply(self):
        d = 89159128863034313675150798691418246016730671603224848136445263738857221457661
        pointRx = 37005820636914709885141514806844168081333130297795324448729113992664980430911
        pointRy = 53341837017606625114352877401388876863372411224581151347982453382597763088853
        pointR = self.pointS * d
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)
        pai = self.pointS.point_at_infinity()
        pointR = self.pointS * 0
        self.assertEqual(pointR, pai)
        self.assertRaises(ValueError, lambda: self.pointS * -1)
        pointR = d * self.pointS
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)
        pointR = Integer(d) * self.pointS
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)

    def test_joing_scalar_multiply(self):
        d = 89159128863034313675150798691418246016730671603224848136445263738857221457661
        e = 95663080849491968089758892602561375436557968422765298393720155425852029574583
        pointRx = 97882805648263813406536008691761336604929644468807939210655781838858762712760
        pointRy = 109601501145939422863091762731388350143838736685237230165293620592272736342645
        pointR = self.pointS * d + self.pointT * e
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)

    def test_sizes(self):
        self.assertEqual(self.pointS.size_in_bits(), 256)
        self.assertEqual(self.pointS.size_in_bytes(), 32)