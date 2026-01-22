from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
class pObj(object):

    def __init__(s, name, t, seq=0):
        s.name = name
        rest = []
        attrs = {}
        for x in t:
            if type(x) == pObj:
                attrs[x.name] = attrs.get(x.name, []) + [x]
            else:
                rest.append(x)
        for name, xs in attrs.items():
            if len(xs) == 1:
                xs = xs[0]
            setattr(s, name, xs)
        s.t = rest
        s.objs = seq and t or []

    def __repr__(s):
        r = []
        for nm in dir(s):
            if nm.startswith('_'):
                continue
            elif nm == 'name':
                continue
            else:
                x = getattr(s, nm)
                if not x:
                    continue
                if type(x) == list_type:
                    r.extend(x)
                else:
                    r.append(x)
        xs = []
        for x in r:
            if isinstance(x, str_type):
                xs.append(x)
            else:
                xs.append(repr(x))
        return '(' + s.name + ' ' + ','.join(xs) + ')'