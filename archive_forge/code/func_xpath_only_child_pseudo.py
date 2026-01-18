import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_only_child_pseudo(self, xpath: XPathExpr) -> XPathExpr:
    return xpath.add_condition('count(parent::*/child::*) = 1')