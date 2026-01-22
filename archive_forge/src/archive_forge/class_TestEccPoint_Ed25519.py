import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import SHAKE128
class TestEccPoint_Ed25519(unittest.TestCase):
    Gxy = {'x': 15112221349535400772501151409588531511454012693041857206046113283949847762202, 'y': 46316835694926478169428394003475163141307993866256225615783033603165251855960}
    G2xy = {'x': 24727413235106541002554574571675588834622768167397638456726423682521233608206, 'y': 15549675580280190176352668710449542251549572066445060580507079593062643049417}
    G3xy = {'x': 46896733464454938657123544595386787789046198280132665686241321779790909858396, 'y': 8324843778533443976490377120369201138301417226297555316741202210403726505172}
    pointG = EccPoint(Gxy['x'], Gxy['y'], curve='Ed25519')
    pointG2 = EccPoint(G2xy['x'], G2xy['y'], curve='Ed25519')
    pointG3 = EccPoint(G3xy['x'], G3xy['y'], curve='Ed25519')

    def test_init_xy(self):
        EccPoint(self.Gxy['x'], self.Gxy['y'], curve='Ed25519')
        pai = EccPoint(0, 1, curve='Ed25519')
        self.assertEqual(pai.x, 0)
        self.assertEqual(pai.y, 1)
        self.assertEqual(pai.xy, (0, 1))
        bp = self.pointG.copy()
        self.assertEqual(bp.x, 15112221349535400772501151409588531511454012693041857206046113283949847762202)
        self.assertEqual(bp.y, 46316835694926478169428394003475163141307993866256225615783033603165251855960)
        self.assertEqual(bp.xy, (bp.x, bp.y))
        bp2 = self.pointG2.copy()
        self.assertEqual(bp2.x, 24727413235106541002554574571675588834622768167397638456726423682521233608206)
        self.assertEqual(bp2.y, 15549675580280190176352668710449542251549572066445060580507079593062643049417)
        self.assertEqual(bp2.xy, (bp2.x, bp2.y))
        EccPoint(x=33467004535436536005251147249499675200073690106659565782908757308821616914995, y=43097193783671926753355113395909008640284023746042808659097434958891230611693, curve='Ed25519')
        self.assertRaises(ValueError, EccPoint, 34, 35, curve='Ed25519')

    def test_set(self):
        pointW = EccPoint(0, 1, curve='Ed25519')
        pointW.set(self.pointG)
        self.assertEqual(pointW.x, self.pointG.x)
        self.assertEqual(pointW.y, self.pointG.y)

    def test_copy(self):
        pointW = self.pointG.copy()
        self.assertEqual(pointW.x, self.pointG.x)
        self.assertEqual(pointW.y, self.pointG.y)

    def test_equal(self):
        pointH = self.pointG.copy()
        pointI = self.pointG2.copy()
        self.assertEqual(self.pointG, pointH)
        self.assertNotEqual(self.pointG, pointI)

    def test_pai(self):
        pai = EccPoint(0, 1, curve='Ed25519')
        self.assertTrue(pai.is_point_at_infinity())
        self.assertEqual(pai, pai.point_at_infinity())

    def test_negate(self):
        negG = -self.pointG
        sum = self.pointG + negG
        self.assertTrue(sum.is_point_at_infinity())

    def test_addition(self):
        self.assertEqual(self.pointG + self.pointG2, self.pointG3)
        self.assertEqual(self.pointG2 + self.pointG, self.pointG3)
        self.assertEqual(self.pointG2 + self.pointG.point_at_infinity(), self.pointG2)
        self.assertEqual(self.pointG.point_at_infinity() + self.pointG2, self.pointG2)
        G5 = self.pointG2 + self.pointG3
        self.assertEqual(G5.x, 33467004535436536005251147249499675200073690106659565782908757308821616914995)
        self.assertEqual(G5.y, 43097193783671926753355113395909008640284023746042808659097434958891230611693)

    def test_inplace_addition(self):
        pointH = self.pointG.copy()
        pointH += self.pointG
        self.assertEqual(pointH, self.pointG2)
        pointH += self.pointG
        self.assertEqual(pointH, self.pointG3)
        pointH += self.pointG.point_at_infinity()
        self.assertEqual(pointH, self.pointG3)

    def test_doubling(self):
        pointH = self.pointG.copy()
        pointH.double()
        self.assertEqual(pointH.x, self.pointG2.x)
        self.assertEqual(pointH.y, self.pointG2.y)
        pai = self.pointG.point_at_infinity()
        pointR = pai.copy()
        pointR.double()
        self.assertEqual(pointR, pai)

    def test_scalar_multiply(self):
        d = 0
        pointH = d * self.pointG
        self.assertEqual(pointH.x, 0)
        self.assertEqual(pointH.y, 1)
        d = 1
        pointH = d * self.pointG
        self.assertEqual(pointH.x, self.pointG.x)
        self.assertEqual(pointH.y, self.pointG.y)
        d = 2
        pointH = d * self.pointG
        self.assertEqual(pointH.x, self.pointG2.x)
        self.assertEqual(pointH.y, self.pointG2.y)
        d = 3
        pointH = d * self.pointG
        self.assertEqual(pointH.x, self.pointG3.x)
        self.assertEqual(pointH.y, self.pointG3.y)
        d = 4
        pointH = d * self.pointG
        self.assertEqual(pointH.x, 14582954232372986451776170844943001818709880559417862259286374126315108956272)
        self.assertEqual(pointH.y, 32483318716863467900234833297694612235682047836132991208333042722294373421359)
        d = 5
        pointH = d * self.pointG
        self.assertEqual(pointH.x, 33467004535436536005251147249499675200073690106659565782908757308821616914995)
        self.assertEqual(pointH.y, 43097193783671926753355113395909008640284023746042808659097434958891230611693)
        d = 10
        pointH = d * self.pointG
        self.assertEqual(pointH.x, 43500613248243327786121022071801015118933854441360174117148262713429272820047)
        self.assertEqual(pointH.y, 45005105423099817237495816771148012388779685712352441364231470781391834741548)
        d = 20
        pointH = d * self.pointG
        self.assertEqual(pointH.x, 46694936775300686710656303283485882876784402425210400817529601134760286812591)
        self.assertEqual(pointH.y, 8786390172762935853260670851718824721296437982862763585171334833968259029560)
        d = 255
        pointH = d * self.pointG
        self.assertEqual(pointH.x, 36843863416400016952258312492144504209624961884991522125275155377549541182230)
        self.assertEqual(pointH.y, 22327030283879720808995671630924669697661065034121040761798775626517750047180)
        d = 256
        pointH = d * self.pointG
        self.assertEqual(pointH.x, 42740085206947573681423002599456489563927820004573071834350074001818321593686)
        self.assertEqual(pointH.y, 6935684722522267618220753829624209639984359598320562595061366101608187623111)

    def test_sizes(self):
        self.assertEqual(self.pointG.size_in_bits(), 255)
        self.assertEqual(self.pointG.size_in_bytes(), 32)