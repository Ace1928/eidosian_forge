from __future__ import print_function
import sys, re
from twisted.words.xish.xpath import AttribValue, BooleanValue, CompareValue
from twisted.words.xish.xpath import Function, IndexValue, LiteralValue
from twisted.words.xish.xpath import _AnyLocation, _Location

XPath Parser.

Besides the parser code produced by Yapps, this module also defines the
parse-time exception classes, a scanner class, a base class for parsers
produced by Yapps, and a context class that keeps track of the parse stack.
These have been copied from the Yapps runtime module.
