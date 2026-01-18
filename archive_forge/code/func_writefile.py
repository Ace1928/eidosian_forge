from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def writefile(pad, fnm, fnmNum, xmldoc, mxlOpt, tOpt=False):
    ipad, ifnm = os.path.split(fnm)
    if tOpt:
        x = xmldoc.findtext('work/work-title', 'no_title')
        ifnm = x.replace(',', '_').replace("'", '_').replace('?', '_')
    else:
        ifnm += fnmNum
    xmlstr = fixDoctype(xmldoc)
    if pad:
        if not mxlOpt or mxlOpt in ['a', 'add']:
            outfnm = os.path.join(pad, ifnm + '.xml')
            outfile = open(outfnm, 'w')
            outfile.write(xmlstr)
            outfile.close()
            info('%s written' % outfnm, warn=0)
        if mxlOpt:
            xml2mxl(pad, ifnm, xmlstr)
    else:
        outfile = sys.stdout
        outfile.write(xmlstr)
        outfile.write('\n')