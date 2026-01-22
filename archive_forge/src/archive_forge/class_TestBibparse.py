import unittest
from pyparsing import ParseException
from .btpyparse import Macro
from . import btpyparse as bp
class TestBibparse(unittest.TestCase):

    def test_names(self):
        bad_chars = '"#%\'(),={}'
        for name_type, dig1f in ((bp.macro_def, False), (bp.field_name, False), (bp.entry_type, False), (bp.cite_key, True)):
            if dig1f:
                self.assertEqual(name_type.parseString('2t')[0], '2t')
            else:
                self.assertRaises(ParseException, name_type.parseString, '2t')
            for char in bad_chars:
                self.assertRaises(ParseException, name_type.parseString, char)
            self.assertEqual(name_type.parseString('simple_test')[0], 'simple_test')
        mr = bp.macro_ref
        self.assertRaises(ParseException, mr.parseString, '2t')
        for char in bad_chars:
            self.assertRaises(ParseException, mr.parseString, char)
        self.assertEqual(mr.parseString('simple_test')[0].name, 'simple_test')

    def test_numbers(self):
        self.assertEqual(bp.number.parseString('1066')[0], '1066')
        self.assertEqual(bp.number.parseString('0')[0], '0')
        self.assertRaises(ParseException, bp.number.parseString, '-4')
        self.assertRaises(ParseException, bp.number.parseString, '+4')
        self.assertRaises(ParseException, bp.number.parseString, '.4')
        self.assertEqual(bp.number.parseString('0.4')[0], '0')

    def test_parse_string(self):
        self.assertEqual(bp.chars_no_quotecurly.parseString('x')[0], 'x')
        self.assertEqual(bp.chars_no_quotecurly.parseString('a string')[0], 'a string')
        self.assertEqual(bp.chars_no_quotecurly.parseString('a "string')[0], 'a ')
        self.assertEqual(bp.chars_no_curly.parseString('x')[0], 'x')
        self.assertEqual(bp.chars_no_curly.parseString('a string')[0], 'a string')
        self.assertEqual(bp.chars_no_curly.parseString('a {string')[0], 'a ')
        self.assertEqual(bp.chars_no_curly.parseString('a }string')[0], 'a ')
        for obj in (bp.curly_string, bp.string, bp.field_value):
            self.assertEqual(obj.parseString('{}').asList(), [])
            self.assertEqual(obj.parseString('{a "string}')[0], 'a "string')
            self.assertEqual(obj.parseString('{a {nested} string}').asList(), ['a ', ['nested'], ' string'])
            self.assertEqual(obj.parseString('{a {double {nested}} string}').asList(), ['a ', ['double ', ['nested']], ' string'])
        for obj in (bp.quoted_string, bp.string, bp.field_value):
            self.assertEqual(obj.parseString('""').asList(), [])
            self.assertEqual(obj.parseString('"a string"')[0], 'a string')
            self.assertEqual(obj.parseString('"a {nested} string"').asList(), ['a ', ['nested'], ' string'])
            self.assertEqual(obj.parseString('"a {double {nested}} string"').asList(), ['a ', ['double ', ['nested']], ' string'])
        self.assertEqual(bp.string.parseString('someascii')[0], Macro('someascii'))
        self.assertRaises(ParseException, bp.string.parseString, '%#= validstring')
        self.assertEqual(bp.string.parseString('1994')[0], '1994')

    def test_parse_field(self):
        fv = bp.field_value
        self.assertEqual(fv.parseString('aname')[0], Macro('aname'))
        self.assertEqual(fv.parseString('ANAME')[0], Macro('aname'))
        self.assertEqual(fv.parseString('aname # "some string"').asList(), [Macro('aname'), 'some string'])
        self.assertEqual(fv.parseString('aname # {some {string}}').asList(), [Macro('aname'), 'some ', ['string']])
        self.assertEqual(fv.parseString('"a string" # 1994').asList(), ['a string', '1994'])
        self.assertEqual(fv.parseString('"a string" # 1994 # a_macro').asList(), ['a string', '1994', Macro('a_macro')])

    def test_comments(self):
        res = bp.comment.parseString('@Comment{about something}')
        self.assertEqual(res.asList(), ['comment', '{about something}'])
        self.assertEqual(bp.comment.parseString('@COMMENT{about something').asList(), ['comment', '{about something'])
        self.assertEqual(bp.comment.parseString('@comment(about something').asList(), ['comment', '(about something'])
        self.assertEqual(bp.comment.parseString('@COMment about something').asList(), ['comment', ' about something'])
        self.assertRaises(ParseException, bp.comment.parseString, '@commentabout something')
        self.assertRaises(ParseException, bp.comment.parseString, '@comment+about something')
        self.assertRaises(ParseException, bp.comment.parseString, '@comment"about something')

    def test_preamble(self):
        res = bp.preamble.parseString('@preamble{"about something"}')
        self.assertEqual(res.asList(), ['preamble', 'about something'])
        self.assertEqual(bp.preamble.parseString('@PREamble{{about something}}').asList(), ['preamble', 'about something'])
        self.assertEqual(bp.preamble.parseString('@PREamble{\n            {about something}\n        }').asList(), ['preamble', 'about something'])

    def test_macro(self):
        res = bp.macro.parseString('@string{ANAME = "about something"}')
        self.assertEqual(res.asList(), ['string', 'aname', 'about something'])
        self.assertEqual(bp.macro.parseString('@string{aname = {about something}}').asList(), ['string', 'aname', 'about something'])

    def test_entry(self):
        txt = '@some_entry{akey, aname = "about something",\n        another={something else}}'
        res = bp.entry.parseString(txt)
        self.assertEqual(res.asList(), ['some_entry', 'akey', ['aname', 'about something'], ['another', 'something else']])
        txt = '@SOME_ENTRY{akey, ANAME = "about something",\n        another={something else}}'
        res = bp.entry.parseString(txt)
        self.assertEqual(res.asList(), ['some_entry', 'akey', ['aname', 'about something'], ['another', 'something else']])

    def test_bibfile(self):
        txt = '@some_entry{akey, aname = "about something",\n        another={something else}}'
        res = bp.bibfile.parseString(txt)
        self.assertEqual(res.asList(), [['some_entry', 'akey', ['aname', 'about something'], ['another', 'something else']]])

    def test_bib1(self):
        txt = '\n    Some introductory text\n    (implicit comment)\n\n        @ARTICLE{Brett2002marsbar,\n      author = {Matthew Brett and Jean-Luc Anton and Romain Valabregue and Jean-Baptise\n                Poline},\n      title = {{Region of interest analysis using an SPM toolbox}},\n      journal = {Neuroimage},\n      year = {2002},\n      volume = {16},\n      pages = {1140--1141},\n      number = {2}\n    }\n\n    @some_entry{akey, aname = "about something",\n    another={something else}}\n    '
        res = bp.bibfile.parseString(txt)
        self.assertEqual(len(res), 3)
        res2 = bp.parse_str(txt)
        self.assertEqual(res.asList(), res2.asList())
        res3 = [r.asList()[0] for r, start, end in bp.definitions.scanString(txt)]
        self.assertEqual(res.asList(), res3)