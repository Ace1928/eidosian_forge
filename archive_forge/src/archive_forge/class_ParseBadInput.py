import sys
from parser import parse, MalformedQueryStringError
from builder import build
import unittest
class ParseBadInput(unittest.TestCase):
    """
    Test for exceptions when bad input is provided
    """
    badQueryStrings = ('f&a hair&sectionnew[]=sekcja siatkarska&sectionnew[]=sekcja siatkarska1&sectionnew[]=sekcja siatkarska2', 'f=a hair&sectionnew[=sekcja siatkarska&sectionnew[]=sekcja siatkarska1&sectionnew[]=sekcja siatkarska2', 'packetname==fd&newsectionname=', "packetname=fd&newsectionname=&section[0]['words'][1", 'packetname=fd&newsectionname=&')

    def test_bad_input(self):
        """parse should fail with malformed querystring"""
        for qstr in self.badQueryStrings:
            self.assertRaises(MalformedQueryStringError, parse, qstr, False)