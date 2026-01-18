import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_relation_child_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
    """right is an immediate child of left; select left"""
    return left.join('[./', right, closing_combiner=']')