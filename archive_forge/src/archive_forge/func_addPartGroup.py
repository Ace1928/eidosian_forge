from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def addPartGroup(sym, num):
    pg = E.Element('part-group', number=str(num), type='start')
    addElem(partlist, pg, lev + 1)
    addElemT(pg, 'group-symbol', sym, lev + 2)
    addElemT(pg, 'group-barline', 'yes', lev + 2)