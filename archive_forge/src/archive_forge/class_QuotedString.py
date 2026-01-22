import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class QuotedString(Token):
    """Token for matching strings that are delimited by quoting characters.
    """

    def __init__(self, quoteChar, escChar=None, escQuote=None, multiline=False, unquoteResults=True, endQuoteChar=None):
        """
           Defined with the following parameters:
            - quoteChar - string of one or more characters defining the quote delimiting string
            - escChar - character to escape quotes, typically backslash (default=None)
            - escQuote - special quote sequence to escape an embedded quote string (such as SQL's "" to escape an embedded ") (default=None)
            - multiline - boolean indicating whether quotes can span multiple lines (default=C{False})
            - unquoteResults - boolean indicating whether the matched text should be unquoted (default=C{True})
            - endQuoteChar - string of one or more characters defining the end of the quote delimited string (default=C{None} => same as quoteChar)
        """
        super(QuotedString, self).__init__()
        quoteChar = quoteChar.strip()
        if len(quoteChar) == 0:
            warnings.warn('quoteChar cannot be the empty string', SyntaxWarning, stacklevel=2)
            raise SyntaxError()
        if endQuoteChar is None:
            endQuoteChar = quoteChar
        else:
            endQuoteChar = endQuoteChar.strip()
            if len(endQuoteChar) == 0:
                warnings.warn('endQuoteChar cannot be the empty string', SyntaxWarning, stacklevel=2)
                raise SyntaxError()
        self.quoteChar = quoteChar
        self.quoteCharLen = len(quoteChar)
        self.firstQuoteChar = quoteChar[0]
        self.endQuoteChar = endQuoteChar
        self.endQuoteCharLen = len(endQuoteChar)
        self.escChar = escChar
        self.escQuote = escQuote
        self.unquoteResults = unquoteResults
        if multiline:
            self.flags = re.MULTILINE | re.DOTALL
            self.pattern = '%s(?:[^%s%s]' % (re.escape(self.quoteChar), _escapeRegexRangeChars(self.endQuoteChar[0]), escChar is not None and _escapeRegexRangeChars(escChar) or '')
        else:
            self.flags = 0
            self.pattern = '%s(?:[^%s\\n\\r%s]' % (re.escape(self.quoteChar), _escapeRegexRangeChars(self.endQuoteChar[0]), escChar is not None and _escapeRegexRangeChars(escChar) or '')
        if len(self.endQuoteChar) > 1:
            self.pattern += '|(?:' + ')|(?:'.join(('%s[^%s]' % (re.escape(self.endQuoteChar[:i]), _escapeRegexRangeChars(self.endQuoteChar[i])) for i in range(len(self.endQuoteChar) - 1, 0, -1))) + ')'
        if escQuote:
            self.pattern += '|(?:%s)' % re.escape(escQuote)
        if escChar:
            self.pattern += '|(?:%s.)' % re.escape(escChar)
            charset = ''.join(set(self.quoteChar[0] + self.endQuoteChar[0])).replace('^', '\\^').replace('-', '\\-')
            self.escCharReplacePattern = re.escape(self.escChar) + '([%s])' % charset
        self.pattern += ')*%s' % re.escape(self.endQuoteChar)
        try:
            self.re = re.compile(self.pattern, self.flags)
            self.reString = self.pattern
        except sre_constants.error:
            warnings.warn('invalid pattern (%s) passed to Regex' % self.pattern, SyntaxWarning, stacklevel=2)
            raise
        self.name = _ustr(self)
        self.errmsg = 'Expected ' + self.name
        self.mayIndexError = False
        self.mayReturnEmpty = True

    def parseImpl(self, instring, loc, doActions=True):
        result = instring[loc] == self.firstQuoteChar and self.re.match(instring, loc) or None
        if not result:
            raise ParseException(instring, loc, self.errmsg, self)
        loc = result.end()
        ret = result.group()
        if self.unquoteResults:
            ret = ret[self.quoteCharLen:-self.endQuoteCharLen]
            if isinstance(ret, basestring):
                if self.escChar:
                    ret = re.sub(self.escCharReplacePattern, '\\g<1>', ret)
                if self.escQuote:
                    ret = ret.replace(self.escQuote, self.endQuoteChar)
        return (loc, ret)

    def __str__(self):
        try:
            return super(QuotedString, self).__str__()
        except:
            pass
        if self.strRepr is None:
            self.strRepr = 'quoted string, starting with %s ending with %s' % (self.quoteChar, self.endQuoteChar)
        return self.strRepr