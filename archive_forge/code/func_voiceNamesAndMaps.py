from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def voiceNamesAndMaps(s, ps):
    vdefs = {}
    for vid, vcedef, vce in ps:
        pname, psubnm = ('', '')
        if not vcedef:
            vdefs[vid] = (pname, psubnm, '')
        else:
            if vid != vcedef.t[1]:
                info('voice ids unequal: %s (reg-ex) != %s (grammar)' % (vid, vcedef.t[1]))
            rn = re.search('(?:name|nm)="([^"]*)"', vcedef.t[2])
            if rn:
                pname = rn.group(1)
            rn = re.search('(?:subname|snm|sname)="([^"]*)"', vcedef.t[2])
            if rn:
                psubnm = rn.group(1)
            vcedef.t[2] = vcedef.t[2].replace('"%s"' % pname, '""').replace('"%s"' % psubnm, '""')
            vdefs[vid] = (pname, psubnm, vcedef.t[2])
        xs = [pObj.t[1] for maat in vce for pObj in maat if pObj.name == 'inline']
        s.staveDefs += [x.replace('%5d', ']') for x in xs if x.startswith('score') or x.startswith('staves')]
    return vdefs