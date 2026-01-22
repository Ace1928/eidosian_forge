import sys
from parser import parse, MalformedQueryStringError
from builder import build
import unittest
class KnownValues(unittest.TestCase):
    """
    Test output for known query string values
    """
    knownValuesClean = ({u'omg': {0: u'0001212'}}, {u'packetname': u'fd', u'section': {0: {u'words': {0: [u'sdfsd', u'ds'], 1: [u'', u''], 2: [u'', u'']}}}, u'language': u'1', u'packetdesc': u'sdfsd', u'newlanguage': u'proponowany jezyk..', u'newsectionname': u'', u'packettype': u'radio'}, {u'packetdesc': u'Zajebiste slowka na jutrzejszy sprawdzian z chemii', u'packetid': u'4', u'packetname': u'Chemia spr', u'section': {10: {u'words': {-1: [u'', u''], -2: [u'', u''], 30: [u'noga', u'leg']}, u'name': u'sekcja siatkarska'}, 11: {u'words': {-1: [u'', u''], -2: [u'', u''], 31: [u'renca', u'rukka']}, u'del_words': {32: [u'kciuk', u'thimb'], 33: [u'oko', u'an eye']}, u'name': u'sekcja siatkarska1'}, 12: {u'words': {-1: [u'', u''], -2: [u'', u''], 34: [u'wlos', u'a hair']}, u'name': u'sekcja siatkarska2'}}, u'language': u'1', u'newlanguage': u'proponowany jezyk..', u'packettype': u'1', u'tags': u'dance, angielski, taniec', u'newsectionname': u'', u'sectionnew': [u'sekcja siatkarska', u'sekcja siatkarska1', u'sekcja siatkarska2']}, {u'f': u'a hair', u'sectionnew': {u'': [u'sekcja siatkarska', u'sekcja siatkarska1', u'sekcja siatkarska2']}}, {u'f': u'a'}, {})
    knownValues = ({u'omg': {0: u'0001212'}}, {u'packetname': u'f&d', u'section': {0: {u'words': {0: [u'sdfsd', u'ds'], 1: [u'', u''], 2: [u'', u'']}}}, u'language': u'1', u'packetdesc': u'sdfsd', u'newlanguage': u'proponowany jezyk..', u'newsectionname': u'', u'packettype': u'radio'}, {u'packetdesc': u'Zajebiste slowka na jutrzejszy sprawdzian z chemii', u'packetid': u'4', u'packetname': u'Chemia spr', u'section': {10: {u'words': {-1: [u'', u''], -2: [u'', u''], 30: [u'noga', u'leg']}, u'name': u'sekcja siatkarska'}, 11: {u'words': {-1: [u'', u''], -2: [u'', u''], 31: [u'renca', u'rukka']}, u'del_words': {32: [u'kciuk', u'thimb'], 33: [u'oko', u'an eye']}, u'name': u'sekcja siatkarska1'}, 12: {u'words': {-1: [u'', u''], -2: [u'', u''], 34: [u'wlos', u'a hair']}, u'name': u'sekcja siatkarska2'}}, u'language': u'1', u'newlanguage': u'proponowany jezyk..', u'packettype': u'1', u'tags': u'dance, angielski, taniec', u'newsectionname': '', u'sectionnew': [u'sekcja=siatkarska', u'sekcja siatkarska1', u'sekcja siatkarska2']}, {u'f': u'a hair', u'sectionnew': {u'': [u'sekcja siatkarska', u'sekcja siatkarska1', u'sekcja siatkarska2']}}, {u'f': u'a'}, {})
    knownValuesCleanWithUnicode = ({u'f': u'逗'},)
    knownValuesWithUnicode = ({u'f': u'逗'},)

    def test_parse_known_values_clean(self):
        """parse should give known result with known input"""
        self.maxDiff = None
        for dic in self.knownValuesClean:
            result = parse(build(dic), unquote=True)
            self.assertEqual(dic, result)

    def test_parse_known_values(self):
        """parse should give known result with known input (quoted)"""
        self.maxDiff = None
        for dic in self.knownValues:
            result = parse(build(dic))
            self.assertEqual(dic, result)

    def test_parse_known_values_clean_with_unicode(self):
        """parse should give known result with known input"""
        self.maxDiff = None
        encoding = 'utf-8' if sys.version_info[0] == 2 else None
        for dic in self.knownValuesClean + self.knownValuesCleanWithUnicode:
            result = parse(build(dic, encoding=encoding), unquote=True, encoding=encoding)
            self.assertEqual(dic, result)

    def test_parse_known_values_with_unicode(self):
        """parse should give known result with known input (quoted)"""
        self.maxDiff = None
        encoding = 'utf-8' if sys.version_info[0] == 2 else None
        for dic in self.knownValues + self.knownValuesWithUnicode:
            result = parse(build(dic, encoding=encoding), encoding=encoding)
            self.assertEqual(dic, result)

    def test_parse_unicode_input_string(self):
        """https://github.com/bernii/querystring-parser/issues/15"""
        qs = u'first_name=%D8%B9%D9%84%DB%8C'
        expected = {u'first_name': u'علی'}
        self.assertEqual(parse(qs.encode('ascii')), expected)
        self.assertEqual(parse(qs), expected)