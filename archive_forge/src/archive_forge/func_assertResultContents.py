import itertools
import logging; log = logging.getLogger(__name__)
from passlib.tests.utils import TestCase
from passlib.pwd import genword, default_charsets
from passlib.pwd import genphrase
def assertResultContents(self, results, count, words, unique=True, sep=' '):
    """check result list matches expected count & charset"""
    self.assertEqual(len(results), count)
    if unique:
        if unique is True:
            unique = count
        self.assertEqual(len(set(results)), unique)
    out = set(itertools.chain.from_iterable((elem.split(sep) for elem in results)))
    self.assertEqual(out, set(words))