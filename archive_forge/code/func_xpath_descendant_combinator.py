import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_descendant_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
    """right is a child, grand-child or further descendant of left"""
    return left.join('/descendant-or-self::*/', right)