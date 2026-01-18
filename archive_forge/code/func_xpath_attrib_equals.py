import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_attrib_equals(self, xpath: XPathExpr, name: str, value: Optional[str]) -> XPathExpr:
    assert value is not None
    xpath.add_condition('%s = %s' % (name, self.xpath_literal(value)))
    return xpath