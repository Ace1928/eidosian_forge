import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
class TestEccPoint_NIST_P521(unittest.TestCase):
    """Tests defined in section 4.5 of https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.204.9073&rep=rep1&type=pdf"""
    pointS = EccPoint(6298662291797597455690424015484694729934739924258269207282256796654663457328131424431164751951136256517278028595707590389210532306396886103392981032134623668, 4353738610347946608426792277824204199293251581155193185744917590170187881415595621178782174392578394291724427970699014037986967831774135041879292583939676195, 'p521')
    pointT = EccPoint(3272445144786979836832049713799473851521209570676705985154696054115223131668939609067329419928734309784154532970182349252877987436515477621030846798794115029, 6837576647799316067119330719965770347825547349379157933006194905270851670833233865575820692115600110447604635634878589709462914482426166204710583103356527365, 'p521')

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
        pointRx = 3945817275303372835061611517163695661138751516365878167904343228042357207513499966884979304221092323287101958990319287179954720158124825676610075596398909929
        pointRy = 1322857172946374883190425618771653850599399337891682578100121118402276271579254324948569667898676732405000107500425170289495924816009002875026615661620603989
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
        pointRx = 3945817275303372835061611517163695661138751516365878167904343228042357207513499966884979304221092323287101958990319287179954720158124825676610075596398909929
        pointRy = 1322857172946374883190425618771653850599399337891682578100121118402276271579254324948569667898676732405000107500425170289495924816009002875026615661620603989
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
        pointRx = 3975062381064225480574098685521977536038322097599329938779879404933190756769282040349324142327608943288400218065083035492863386391810636410549184121094237126
        pointRy = 5475657578453756240260133840068900684884763004547062714625687949872864918166460799367981360733022202781327245597758551064385199362201914511669512056775423299
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
        d = 6589911711217983778573069477633357728158042172680965023176544155025330858580714983813612606791942793065205275863059605874060770880450379314361467079894228145
        pointRx = 1953421426470442621175127908579910595115753295475953441048167754517888749470193515243737962476875375071476192850873606070916658618742628469507462988779263419
        pointRy = 4089013307542931896517959004042561972312454331747019008599426079478882447228510735462753012916267465837819004434222633127652386667671808349966574478000887679
        pointR = self.pointS * d
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)
        pai = self.pointS.point_at_infinity()
        pointR = self.pointS * 0
        self.assertEqual(pointR, pai)
        self.assertRaises(ValueError, lambda: self.pointS * -1)

    def test_joing_scalar_multiply(self):
        d = 6589911711217983778573069477633357728158042172680965023176544155025330858580714983813612606791942793065205275863059605874060770880450379314361467079894228145
        e = 4181911831981269416356172644541385942225844343101932797409015214375313229505212523787171269748633152878588298057590237947985265681987190725291258965278331619
        pointRx = 2107959292211217366325006629500797925791938696205964597056372330716592288740742242015151987654379328611137342287481965102852833591184243752401697874676180285
        pointRy = 213072210276359788783200291437892703727051498421519023277575236351664969296409578866338818293271364049979364487784950198387335086007766121902297674599028893
        pointR = self.pointS * d
        pointR += self.pointT * e
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)

    def test_sizes(self):
        self.assertEqual(self.pointS.size_in_bits(), 521)
        self.assertEqual(self.pointS.size_in_bytes(), 66)