import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_attrib(self, selector: Attrib) -> XPathExpr:
    """Translate an attribute selector."""
    operator = self.attribute_operator_mapping[selector.operator]
    method = getattr(self, 'xpath_attrib_%s' % operator)
    if self.lower_case_attribute_names:
        name = selector.attrib.lower()
    else:
        name = selector.attrib
    safe = is_safe_name(name)
    if selector.namespace:
        name = '%s:%s' % (selector.namespace, name)
        safe = safe and is_safe_name(selector.namespace)
    if safe:
        attrib = '@' + name
    else:
        attrib = 'attribute::*[name() = %s]' % self.xpath_literal(name)
    if selector.value is None:
        value = None
    elif self.lower_case_attribute_values:
        value = typing.cast(str, selector.value.value).lower()
    else:
        value = selector.value.value
    return typing.cast(XPathExpr, method(self.xpath(selector.selector), attrib, value))