import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_relation(self, relation: Relation) -> XPathExpr:
    xpath = self.xpath(relation.selector)
    combinator = relation.combinator
    subselector = relation.subselector
    right = self.xpath(subselector.parsed_tree)
    method = getattr(self, 'xpath_relation_%s_combinator' % self.combinator_mapping[typing.cast(str, combinator.value)])
    return typing.cast(XPathExpr, method(xpath, right))