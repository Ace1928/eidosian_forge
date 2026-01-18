from itertools import chain
import re
from tokenize import PseudoToken
from genshi.core import TEXT
from genshi.template.base import TemplateSyntaxError, EXPR
from genshi.template.eval import Expression
Parse the given string and extract expressions.
    
    This function is a generator that yields `TEXT` events for literal strings,
    and `EXPR` events for expressions, depending on the results of parsing the
    string.
    
    >>> for kind, data, pos in interpolate("hey ${foo}bar"):
    ...     print('%s %r' % (kind, data))
    TEXT 'hey '
    EXPR Expression('foo')
    TEXT 'bar'
    
    :param text: the text to parse
    :param filepath: absolute path to the file in which the text was found
                     (optional)
    :param lineno: the line number at which the text was found (optional)
    :param offset: the column number at which the text starts in the source
                   (optional)
    :param lookup: the variable lookup mechanism; either "lenient" (the
                   default), "strict", or a custom lookup class
    :return: a list of `TEXT` and `EXPR` events
    :raise TemplateSyntaxError: when a syntax error in an expression is
                                encountered
    