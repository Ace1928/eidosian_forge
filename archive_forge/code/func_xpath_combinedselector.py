import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_combinedselector(self, combined: CombinedSelector) -> XPathExpr:
    """Translate a combined selector."""
    combinator = self.combinator_mapping[combined.combinator]
    method = getattr(self, 'xpath_%s_combinator' % combinator)
    return typing.cast(XPathExpr, method(self.xpath(combined.selector), self.xpath(combined.subselector)))