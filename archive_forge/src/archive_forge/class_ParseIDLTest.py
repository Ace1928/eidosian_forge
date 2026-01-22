from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseIDLTest(ParseTestCase):

    def runTest(self):
        from examples import idlParse

        def test(strng, numToks, errloc=0):
            print_(strng)
            try:
                bnf = idlParse.CORBA_IDL_BNF()
                tokens = bnf.parseString(strng)
                print_('tokens = ')
                tokens.pprint()
                tokens = flatten(tokens.asList())
                print_(len(tokens))
                self.assertEqual(len(tokens), numToks, 'error matching IDL string, %s -> %s' % (strng, str(tokens)))
            except ParseException as err:
                print_(err.line)
                print_(' ' * (err.column - 1) + '^')
                print_(err)
                self.assertEqual(numToks, 0, 'unexpected ParseException while parsing %s, %s' % (strng, str(err)))
                self.assertEqual(err.loc, errloc, 'expected ParseException at %d, found exception at %d' % (errloc, err.loc))
        test('\n            /*\n             * a block comment *\n             */\n            typedef string[10] tenStrings;\n            typedef sequence<string> stringSeq;\n            typedef sequence< sequence<string> > stringSeqSeq;\n\n            interface QoSAdmin {\n                stringSeq method1( in string arg1, inout long arg2 );\n                stringSeqSeq method2( in string arg1, inout long arg2, inout long arg3);\n                string method3();\n              };\n            ', 59)
        test('\n            /*\n             * a block comment *\n             */\n            typedef string[10] tenStrings;\n            typedef\n                /** ** *** **** *\n                 * a block comment *\n                 */\n                sequence<string> /*comment inside an And */ stringSeq;\n            /* */  /**/ /***/ /****/\n            typedef sequence< sequence<string> > stringSeqSeq;\n\n            interface QoSAdmin {\n                stringSeq method1( in string arg1, inout long arg2 );\n                stringSeqSeq method2( in string arg1, inout long arg2, inout long arg3);\n                string method3();\n              };\n            ', 59)
        test('\n              const string test="Test String\\n";\n              const long  a = 0;\n              const long  b = -100;\n              const float c = 3.14159;\n              const long  d = 0x007f7f7f;\n              exception TestException\n                {\n                string msg;\n                sequence<string> dataStrings;\n                };\n\n              interface TestInterface\n                {\n                void method1( in string arg1, inout long arg2 );\n                };\n            ', 60)
        test('\n            module Test1\n              {\n              exception TestException\n                {\n                string msg;\n                ];\n\n              interface TestInterface\n                {\n                void method1( in string arg1, inout long arg2 )\n                  raises ( TestException );\n                };\n              };\n            ', 0, 56)
        test('\n            module Test1\n              {\n              exception TestException\n                {\n                string msg;\n                };\n\n              };\n            ', 13)