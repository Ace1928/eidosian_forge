import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import SHAKE128
class TestEccPoint_Ed448(unittest.TestCase):
    Gxy = {'x': 224580040295924300187604334099896036246789641632564134246125461686950415467406032909029192869357953282578032075146446173674602635247710, 'y': 298819210078481492676017930443930673437544040154080242095928241372331506189835876003536878655418784733982303233503462500531545062832660}
    G2xy = {'x': 484559149530404593699549205258669689569094240458212040187660132787056912146709081364401144455726350866276831544947397859048262938744149, 'y': 494088759867433727674302672526735089350544552303727723746126484473087719117037293890093462157703888342865036477787453078312060500281069}
    G3xy = {'x': 23839778817283171003887799738662344287085130522697782688245073320169861206004018274567429238677677920280078599146891901463786155880335, 'y': 636046652612779686502873775776967954190574036985351036782021535703553242737829645273154208057988851307101009474686328623630835377952508}
    pointG = EccPoint(Gxy['x'], Gxy['y'], curve='Ed448')
    pointG2 = EccPoint(G2xy['x'], G2xy['y'], curve='Ed448')
    pointG3 = EccPoint(G3xy['x'], G3xy['y'], curve='Ed448')

    def test_init_xy(self):
        EccPoint(self.Gxy['x'], self.Gxy['y'], curve='Ed448')
        pai = EccPoint(0, 1, curve='Ed448')
        self.assertEqual(pai.x, 0)
        self.assertEqual(pai.y, 1)
        self.assertEqual(pai.xy, (0, 1))
        bp = self.pointG.copy()
        self.assertEqual(bp.x, 224580040295924300187604334099896036246789641632564134246125461686950415467406032909029192869357953282578032075146446173674602635247710)
        self.assertEqual(bp.y, 298819210078481492676017930443930673437544040154080242095928241372331506189835876003536878655418784733982303233503462500531545062832660)
        self.assertEqual(bp.xy, (bp.x, bp.y))
        bp2 = self.pointG2.copy()
        self.assertEqual(bp2.x, 484559149530404593699549205258669689569094240458212040187660132787056912146709081364401144455726350866276831544947397859048262938744149)
        self.assertEqual(bp2.y, 494088759867433727674302672526735089350544552303727723746126484473087719117037293890093462157703888342865036477787453078312060500281069)
        self.assertEqual(bp2.xy, (bp2.x, bp2.y))
        EccPoint(x=348153875026276389637043645049107892082020142965264203606298947249372502263734068726043414184711136479257951263857196303016452373545012, y=493994996594634813057750972404082576748165641003441402536643796919488930346488936598199156959526972385489216744248713810037470609552875, curve='Ed448')
        self.assertRaises(ValueError, EccPoint, 34, 35, curve='Ed448')

    def test_set(self):
        pointW = EccPoint(0, 1, curve='Ed448')
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
        pai = EccPoint(0, 1, curve='Ed448')
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
        self.assertEqual(G5.x, 348153875026276389637043645049107892082020142965264203606298947249372502263734068726043414184711136479257951263857196303016452373545012)
        self.assertEqual(G5.y, 493994996594634813057750972404082576748165641003441402536643796919488930346488936598199156959526972385489216744248713810037470609552875)

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
        self.assertEqual(pointH.x, 209710714663589237570084264541991420589833663592202160838176801982171960997051286469552065490170659385708816452452440655275673121357616)
        self.assertEqual(pointH.y, 603515570432573637134887094808958022419371301976351441963100315034426774344109511210661998660350679225364893651728492312845104034682937)
        d = 5
        pointH = d * self.pointG
        self.assertEqual(pointH.x, 348153875026276389637043645049107892082020142965264203606298947249372502263734068726043414184711136479257951263857196303016452373545012)
        self.assertEqual(pointH.y, 493994996594634813057750972404082576748165641003441402536643796919488930346488936598199156959526972385489216744248713810037470609552875)
        d = 10
        pointH = d * self.pointG
        self.assertEqual(pointH.x, 338669802554017338090741842335737823113336653904160109397584018105059385206196637853257117971041875208554805602850382915119972512247869)
        self.assertEqual(pointH.y, 219150861381236143450521108354203769270679720218000652369136137699333671138749674479139795638088562049448210382871218396511167516106599)
        d = 20
        pointH = d * self.pointG
        self.assertEqual(pointH.x, 170745337849404729037752671328115757395299314398846059689519355726549085843629977531480785436839092249575248500245151035929423772455446)
        self.assertEqual(pointH.y, 514848433544469854353946688763194387200718510630675324106981980922630260002444912225611122917375511398028927935674718550721459447371489)
        d = 255
        pointH = d * self.pointG
        self.assertEqual(pointH.x, 541490963568107027170705286897170109510461691773110357043636717804671871894892366229727162930291781497782561357264712160149279167748479)
        self.assertEqual(pointH.y, 85788530468533985949094930687552878336961725667667139455803222592485001115890239219551094621532028384006276147158803896215486675922259)
        d = 256
        pointH = d * self.pointG
        self.assertEqual(pointH.x, 685982959581589945797259192133208736810081129344694610386175603458136735629156952501268376831528231173577780418249504359009186849205276)
        self.assertEqual(pointH.y, 170124505282579497867188709538884189011491629827746027149982848266301129567126662749084431227442982360871397078716303280358380463465502)

    def test_sizes(self):
        self.assertEqual(self.pointG.size_in_bits(), 448)
        self.assertEqual(self.pointG.size_in_bytes(), 56)