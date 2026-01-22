from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def addDirection(parent, elems, lev, gstaff, subelms=[], placement='below', cue_on=0):
    dir = E.Element('direction', placement=placement)
    addElem(parent, dir, lev)
    if type(elems) != list_type:
        elems = [(elems, subelms)]
    for elem, subelms in elems:
        typ = E.Element('direction-type')
        addElem(dir, typ, lev + 1)
        addElem(typ, elem, lev + 2)
        for subel in subelms:
            addElem(elem, subel, lev + 3)
    if cue_on:
        addElem(dir, E.Element('level', size='cue'), lev + 1)
    if gstaff:
        addElemT(dir, 'staff', str(gstaff), lev + 1)
    return dir