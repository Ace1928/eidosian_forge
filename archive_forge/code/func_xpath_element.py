import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_element(self, selector: Element) -> XPathExpr:
    """Translate a type or universal selector."""
    element = selector.element
    if not element:
        element = '*'
        safe = True
    else:
        safe = bool(is_safe_name(element))
        if self.lower_case_element_names:
            element = element.lower()
    if selector.namespace:
        element = '%s:%s' % (selector.namespace, element)
        safe = safe and bool(is_safe_name(selector.namespace))
    xpath = self.xpathexpr_cls(element=element)
    if not safe:
        xpath.add_name_test()
    return xpath