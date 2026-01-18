import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_attrib_suffixmatch(self, xpath: XPathExpr, name: str, value: Optional[str]) -> XPathExpr:
    if value:
        xpath.add_condition('%s and substring(%s, string-length(%s)-%s) = %s' % (name, name, name, len(value) - 1, self.xpath_literal(value)))
    else:
        xpath.add_condition('0')
    return xpath