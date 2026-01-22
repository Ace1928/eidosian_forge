import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class Each(ParseExpression):
    """Requires all given C{ParseExpression}s to be found, but in any order.
       Expressions may be separated by whitespace.
       May be constructed using the C{'&'} operator.
    """

    def __init__(self, exprs, savelist=True):
        super(Each, self).__init__(exprs, savelist)
        self.mayReturnEmpty = True
        for e in self.exprs:
            if not e.mayReturnEmpty:
                self.mayReturnEmpty = False
                break
        self.skipWhitespace = True
        self.initExprGroups = True

    def parseImpl(self, instring, loc, doActions=True):
        if self.initExprGroups:
            opt1 = [e.expr for e in self.exprs if isinstance(e, Optional)]
            opt2 = [e for e in self.exprs if e.mayReturnEmpty and e not in opt1]
            self.optionals = opt1 + opt2
            self.multioptionals = [e.expr for e in self.exprs if isinstance(e, ZeroOrMore)]
            self.multirequired = [e.expr for e in self.exprs if isinstance(e, OneOrMore)]
            self.required = [e for e in self.exprs if not isinstance(e, (Optional, ZeroOrMore, OneOrMore))]
            self.required += self.multirequired
            self.initExprGroups = False
        tmpLoc = loc
        tmpReqd = self.required[:]
        tmpOpt = self.optionals[:]
        matchOrder = []
        keepMatching = True
        while keepMatching:
            tmpExprs = tmpReqd + tmpOpt + self.multioptionals + self.multirequired
            failed = []
            for e in tmpExprs:
                try:
                    tmpLoc = e.tryParse(instring, tmpLoc)
                except ParseException:
                    failed.append(e)
                else:
                    matchOrder.append(e)
                    if e in tmpReqd:
                        tmpReqd.remove(e)
                    elif e in tmpOpt:
                        tmpOpt.remove(e)
            if len(failed) == len(tmpExprs):
                keepMatching = False
        if tmpReqd:
            missing = ', '.join((_ustr(e) for e in tmpReqd))
            raise ParseException(instring, loc, 'Missing one or more required elements (%s)' % missing)
        matchOrder += [e for e in self.exprs if isinstance(e, Optional) and e.expr in tmpOpt]
        resultlist = []
        for e in matchOrder:
            loc, results = e._parse(instring, loc, doActions)
            resultlist.append(results)
        finalResults = ParseResults([])
        for r in resultlist:
            dups = {}
            for k in r.keys():
                if k in finalResults.keys():
                    tmp = ParseResults(finalResults[k])
                    tmp += ParseResults(r[k])
                    dups[k] = tmp
            finalResults += ParseResults(r)
            for k, v in dups.items():
                finalResults[k] = v
        return (loc, finalResults)

    def __str__(self):
        if hasattr(self, 'name'):
            return self.name
        if self.strRepr is None:
            self.strRepr = '{' + ' & '.join((_ustr(e) for e in self.exprs)) + '}'
        return self.strRepr

    def checkRecursion(self, parseElementList):
        subRecCheckList = parseElementList[:] + [self]
        for e in self.exprs:
            e.checkRecursion(subRecCheckList)