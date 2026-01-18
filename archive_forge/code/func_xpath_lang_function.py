import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_lang_function(self, xpath: XPathExpr, function: Function) -> XPathExpr:
    if function.argument_types() not in (['STRING'], ['IDENT']):
        raise ExpressionError('Expected a single string or ident for :lang(), got %r' % function.arguments)
    value = function.arguments[0].value
    assert value
    return xpath.add_condition("ancestor-or-self::*[@lang][1][starts-with(concat(translate(@%s, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '-'), %s)]" % (self.lang_attribute, self.xpath_literal(value.lower() + '-')))