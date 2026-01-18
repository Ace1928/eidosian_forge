from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def xml2mxl(pad, fnm, data):
    from zipfile import ZipFile, ZIP_DEFLATED
    fnmext = fnm + '.xml'
    outfile = os.path.join(pad, fnm + '.mxl')
    meta = '%s\n<container><rootfiles>\n' % xmlVersion
    meta += '<rootfile full-path="%s" media-type="application/vnd.recordare.musicxml+xml"/>\n' % fnmext
    meta += '</rootfiles></container>'
    f = ZipFile(outfile, 'w', ZIP_DEFLATED)
    f.writestr('META-INF/container.xml', meta)
    f.writestr(fnmext, data)
    f.close()
    info('%s written' % outfile, warn=0)