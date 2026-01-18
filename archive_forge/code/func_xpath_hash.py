import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_hash(self, id_selector: Hash) -> XPathExpr:
    """Translate an ID selector."""
    xpath = self.xpath(id_selector.selector)
    return self.xpath_attrib_equals(xpath, '@id', id_selector.id)