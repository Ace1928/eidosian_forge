from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ReStringRangeTest(ParseTestCase):

    def runTest(self):
        testCases = ('[A-Z]', '[A-A]', '[A-Za-z]', '[A-z]', '[\\ -\\~]', '[\\0x20-0]', '[\\0x21-\\0x7E]', '[\\0xa1-\\0xfe]', '[\\040-0]', '[A-Za-z0-9]', '[A-Za-z0-9_]', '[A-Za-z0-9_$]', '[A-Za-z0-9_$\\-]', '[^0-9\\\\]', '[a-zA-Z]', '[/\\^~]', '[=\\+\\-!]', '[A-]', '[-A]', '[\\x21]', u'[а-яА-ЯёЁABCDEFGHIJKLMNOPQRSTUVWXYZ$_!α-ω]')
        expectedResults = ('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'A', 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz', ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~', ' !"#$%&\'()*+,-./0', '!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~', u'¡¢£¤¥¦§¨©ª«¬\xad®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþ', ' !"#$%&\'()*+,-./0', 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_', 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_$', 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_$-', '0123456789\\', 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', '/^~', '=+-!', 'A-', '-A', '!', u'абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯёЁABCDEFGHIJKLMNOPQRSTUVWXYZ$_!αβγδεζηθικλμνξοπρςστυφχψω')
        for test in zip(testCases, expectedResults):
            t, exp = test
            res = pp.srange(t)
            self.assertEqual(res, exp, "srange error, srange(%r)->'%r', expected '%r'" % (t, res, exp))