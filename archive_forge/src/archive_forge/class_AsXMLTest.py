from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class AsXMLTest(ParseTestCase):

    def runTest(self):
        aaa = pp.Word('a')('A')
        bbb = pp.Group(pp.Word('b'))('B')
        ccc = pp.Combine(':' + pp.Word('c'))('C')
        g1 = 'XXX>&<' + pp.ZeroOrMore(aaa | bbb | ccc)
        teststring = 'XXX>&< b b a b b a b :c b a'
        print_('test including all items')
        xml = g1.parseString(teststring).asXML('TEST', namedItemsOnly=False)
        assert xml == '\n'.join(['', '<TEST>', '  <ITEM>XXX&gt;&amp;&lt;</ITEM>', '  <B>', '    <ITEM>b</ITEM>', '  </B>', '  <B>', '    <ITEM>b</ITEM>', '  </B>', '  <A>a</A>', '  <B>', '    <ITEM>b</ITEM>', '  </B>', '  <B>', '    <ITEM>b</ITEM>', '  </B>', '  <A>a</A>', '  <B>', '    <ITEM>b</ITEM>', '  </B>', '  <C>:c</C>', '  <B>', '    <ITEM>b</ITEM>', '  </B>', '  <A>a</A>', '</TEST>']), 'failed to generate XML correctly showing all items: \n[' + xml + ']'
        print_('test filtering unnamed items')
        xml = g1.parseString(teststring).asXML('TEST', namedItemsOnly=True)
        assert xml == '\n'.join(['', '<TEST>', '  <B>', '    <ITEM>b</ITEM>', '  </B>', '  <B>', '    <ITEM>b</ITEM>', '  </B>', '  <A>a</A>', '  <B>', '    <ITEM>b</ITEM>', '  </B>', '  <B>', '    <ITEM>b</ITEM>', '  </B>', '  <A>a</A>', '  <B>', '    <ITEM>b</ITEM>', '  </B>', '  <C>:c</C>', '  <B>', '    <ITEM>b</ITEM>', '  </B>', '  <A>a</A>', '</TEST>']), 'failed to generate XML correctly, filtering unnamed items: ' + xml