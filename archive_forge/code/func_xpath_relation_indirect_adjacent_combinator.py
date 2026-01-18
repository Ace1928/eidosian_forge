import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_relation_indirect_adjacent_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
    """right is a sibling after left, immediately or not; select left"""
    return left.join('[following-sibling::', right, closing_combiner=']')