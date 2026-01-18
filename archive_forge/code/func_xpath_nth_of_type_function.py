import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_nth_of_type_function(self, xpath: XPathExpr, function: Function) -> XPathExpr:
    if xpath.element == '*':
        raise ExpressionError('*:nth-of-type() is not implemented')
    return self.xpath_nth_child_function(xpath, function, add_name_test=False)