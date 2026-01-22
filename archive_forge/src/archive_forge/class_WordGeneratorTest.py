import itertools
import logging; log = logging.getLogger(__name__)
from passlib.tests.utils import TestCase
from passlib.pwd import genword, default_charsets
from passlib.pwd import genphrase
class WordGeneratorTest(TestCase):
    """test generation routines"""
    descriptionPrefix = 'passlib.pwd.genword()'

    def setUp(self):
        super(WordGeneratorTest, self).setUp()
        from passlib.pwd import SequenceGenerator
        self.patchAttr(SequenceGenerator, 'rng', self.getRandom('pwd generator'))

    def assertResultContents(self, results, count, chars, unique=True):
        """check result list matches expected count & charset"""
        self.assertEqual(len(results), count)
        if unique:
            if unique is True:
                unique = count
            self.assertEqual(len(set(results)), unique)
        self.assertEqual(set(''.join(results)), set(chars))

    def test_general(self):
        """general behavior"""
        result = genword()
        self.assertEqual(len(result), 9)
        self.assertRaisesRegex(TypeError, '(?i)unexpected keyword.*badkwd', genword, badkwd=True)

    def test_returns(self):
        """'returns' keyword"""
        results = genword(returns=5000)
        self.assertResultContents(results, 5000, ascii_62)
        gen = genword(returns=iter)
        results = [next(gen) for _ in range(5000)]
        self.assertResultContents(results, 5000, ascii_62)
        self.assertRaises(TypeError, genword, returns='invalid-type')

    def test_charset(self):
        """'charset' & 'chars' options"""
        results = genword(charset='hex', returns=5000)
        self.assertResultContents(results, 5000, hex)
        results = genword(length=3, chars='abc', returns=5000)
        self.assertResultContents(results, 5000, 'abc', unique=27)
        self.assertRaises(TypeError, genword, chars='abc', charset='hex')