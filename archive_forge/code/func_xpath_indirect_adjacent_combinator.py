import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_indirect_adjacent_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
    """right is a sibling after left, immediately or not"""
    return left.join('/following-sibling::', right)