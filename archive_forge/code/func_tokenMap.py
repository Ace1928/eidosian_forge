import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import collections
import pprint
import traceback
import types
from datetime import datetime
def tokenMap(func, *args):
    """
    Helper to define a parse action by mapping a function to all elements of a ParseResults list.If any additional 
    args are passed, they are forwarded to the given function as additional arguments after
    the token, as in C{hex_integer = Word(hexnums).setParseAction(tokenMap(int, 16))}, which will convert the
    parsed data to an integer using base 16.

    Example (compare the last to example in L{ParserElement.transformString}::
        hex_ints = OneOrMore(Word(hexnums)).setParseAction(tokenMap(int, 16))
        hex_ints.runTests('''
            00 11 22 aa FF 0a 0d 1a
            ''')
        
        upperword = Word(alphas).setParseAction(tokenMap(str.upper))
        OneOrMore(upperword).runTests('''
            my kingdom for a horse
            ''')

        wd = Word(alphas).setParseAction(tokenMap(str.title))
        OneOrMore(wd).setParseAction(' '.join).runTests('''
            now is the winter of our discontent made glorious summer by this sun of york
            ''')
    prints::
        00 11 22 aa FF 0a 0d 1a
        [0, 17, 34, 170, 255, 10, 13, 26]

        my kingdom for a horse
        ['MY', 'KINGDOM', 'FOR', 'A', 'HORSE']

        now is the winter of our discontent made glorious summer by this sun of york
        ['Now Is The Winter Of Our Discontent Made Glorious Summer By This Sun Of York']
    """

    def pa(s, l, t):
        return [func(tokn, *args) for tokn in t]
    try:
        func_name = getattr(func, '__name__', getattr(func, '__class__').__name__)
    except Exception:
        func_name = str(func)
    pa.__name__ = func_name
    return pa