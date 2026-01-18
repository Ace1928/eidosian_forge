import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_nth_last_child_function(self, xpath: XPathExpr, function: Function) -> XPathExpr:
    return self.xpath_nth_child_function(xpath, function, last=True)