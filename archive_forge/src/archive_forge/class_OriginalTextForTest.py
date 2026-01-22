from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class OriginalTextForTest(ParseTestCase):

    def runTest(self):
        from pyparsing import makeHTMLTags, originalTextFor

        def rfn(t):
            return '%s:%d' % (t.src, len(''.join(t)))
        makeHTMLStartTag = lambda tag: originalTextFor(makeHTMLTags(tag)[0], asString=False)
        start = makeHTMLStartTag('IMG')
        start.addParseAction(rfn)
        text = '_<img src="images/cal.png"\n            alt="cal image" width="16" height="15">_'
        s = start.transformString(text)
        if VERBOSE:
            print_(s)
        self.assertTrue(s.startswith('_images/cal.png:'), 'failed to preserve input s properly')
        self.assertTrue(s.endswith('77_'), 'failed to return full original text properly')
        tag_fields = makeHTMLStartTag('IMG').searchString(text)[0]
        if VERBOSE:
            print_(sorted(tag_fields.keys()))
            self.assertEqual(sorted(tag_fields.keys()), ['alt', 'empty', 'height', 'src', 'startImg', 'tag', 'width'], 'failed to preserve results names in originalTextFor')