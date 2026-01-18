import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_checked_pseudo(self, xpath: XPathExpr) -> XPathExpr:
    return xpath.add_condition("(@selected and name(.) = 'option') or (@checked and (name(.) = 'input' or name(.) = 'command')and (@type = 'checkbox' or @type = 'radio'))")