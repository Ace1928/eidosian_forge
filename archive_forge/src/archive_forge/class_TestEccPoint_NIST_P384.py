import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
class TestEccPoint_NIST_P384(unittest.TestCase):
    """Tests defined in section 4.4 of https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.204.9073&rep=rep1&type=pdf"""
    pointS = EccPoint(38729843255500722012223789140386576369846945764011622016853073910111100585027271282618686028819155829771986686692110, 14775065806045687447040329012189970143110457192273170002438846587135550607124932692645668896323008136834786699796037, 'p384')
    pointT = EccPoint(26288057065583075782348006106204037622020428415618373022120672060782128318684539495108859203998285343435414872526929, 20317021471476661004231932059921416119243482095142802007888854028692143286703870798313405506767818863582084648882322, 'p384')

    def test_set(self):
        pointW = EccPoint(0, 0, 'p384')
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
        pointRx = 2902941670251247452004328987891566498498418822617372054018910414220003594297713566844355824247021294586603319811821
        pointRy = 3455295380906732143117785965051679429645957923970501876685534267370842144378305510573333094428212006777379199689904
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

    def _test_inplace_addition(self):
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
        pointRx = 6484273686407288417863186493719738906843255438499188335753863058292644924002854404495998886943663447112418995556428
        pointRy = 38506322822213971190402483393098207294760371125428025044673909732301449549436866375736843152912072714143144334482909
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
        d = 25383675053754323265313847600763558649837567523936419382180429049411419453287961533915959958692763000428249340195968
        pointRx = 35241211903848585764193417256089412244077491733824912416779335999929152525681727503289566598393046444215249706487538
        pointRy = 26548935833063912180659367497129494200202067210914187435958482957927711256118273203327173901227150607470767084496511
        pointR = self.pointS * d
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)
        pai = self.pointS.point_at_infinity()
        pointR = self.pointS * 0
        self.assertEqual(pointR, pai)
        self.assertRaises(ValueError, lambda: self.pointS * -1)

    def test_joing_scalar_multiply(self):
        d = 25383675053754323265313847600763558649837567523936419382180429049411419453287961533915959958692763000428249340195968
        e = 27059738705138967694609917441539569198565790438030441179187158346248993390645567164253698976702013943331436929300573
        pointRx = 22393678908170526983358878568961791868296090732839628770081764691104229323619227685842806040476371693656649051538140
        pointRy = 4026998541038046337974532785211329836553503965631023223516815877447140921515225472995530218179372640183526353760646
        pointR = self.pointS * d + self.pointT * e
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)

    def test_sizes(self):
        self.assertEqual(self.pointS.size_in_bits(), 384)
        self.assertEqual(self.pointS.size_in_bytes(), 48)