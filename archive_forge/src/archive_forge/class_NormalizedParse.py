import sys
from parser import parse, MalformedQueryStringError
from builder import build
import unittest
class NormalizedParse(unittest.TestCase):
    """
    """
    knownValues = {'section': {10: {u'words': {-1: [u'', u''], -2: [u'', u''], 30: [u'noga', u'leg']}, 'name': u'sekcja siatkarska'}, 11: {u'words': {-1: [u'', u''], -2: [u'', u''], 31: [u'renca', u'rukka']}, u'del_words': {32: [u'kciuk', u'thimb'], 33: [u'oko', u'an eye']}, u'name': u'sekcja siatkarska1'}, 12: {u'words': {-1: [u'', u''], -2: [u'', u''], 34: [u'wlos', u'a hair']}, u'name': u'sekcja siatkarska2'}}}
    knownValuesNormalized = {'section': [{'name': 'sekcja siatkarska', 'words': [['', ''], ['', ''], ['noga', 'leg']]}, {'del_words': [['kciuk', 'thimb'], ['oko', 'an eye']], 'name': 'sekcja siatkarska1', 'words': [['', ''], ['', ''], ['renca', 'rukka']]}, {'name': 'sekcja siatkarska2', 'words': [['wlos', 'a hair'], ['', ''], ['', '']]}]}

    def test_parse_normalized(self):
        result = parse(build(self.knownValues), normalized=True)
        self.assertEqual(self.knownValuesNormalized, result)