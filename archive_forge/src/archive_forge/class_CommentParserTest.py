from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class CommentParserTest(ParseTestCase):

    def runTest(self):
        print_('verify processing of C and HTML comments')
        testdata = '\n        /* */\n        /** **/\n        /**/\n        /***/\n        /****/\n        /* /*/\n        /** /*/\n        /*** /*/\n        /*\n         ablsjdflj\n         */\n        '
        foundLines = [pp.lineno(s, testdata) for t, s, e in pp.cStyleComment.scanString(testdata)]
        self.assertEqual(foundLines, list(range(11))[2:], 'only found C comments on lines ' + str(foundLines))
        testdata = '\n        <!-- -->\n        <!--- --->\n        <!---->\n        <!----->\n        <!------>\n        <!-- /-->\n        <!--- /-->\n        <!---- /-->\n        <!---- /- ->\n        <!---- / -- >\n        <!--\n         ablsjdflj\n         -->\n        '
        foundLines = [pp.lineno(s, testdata) for t, s, e in pp.htmlComment.scanString(testdata)]
        self.assertEqual(foundLines, list(range(11))[2:], 'only found HTML comments on lines ' + str(foundLines))
        testSource = '\n            // comment1\n            // comment2 \\\n            still comment 2\n            // comment 3\n            '
        self.assertEqual(len(pp.cppStyleComment.searchString(testSource)[1][0]), 41, "failed to match single-line comment with '\\' at EOL")