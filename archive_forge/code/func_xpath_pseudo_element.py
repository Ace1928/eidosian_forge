import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_pseudo_element(self, xpath: XPathExpr, pseudo_element: PseudoElement) -> XPathExpr:
    """Translate a pseudo-element.

        Defaults to not supporting pseudo-elements at all,
        but can be overridden by sub-classes.

        """
    raise ExpressionError('Pseudo-elements are not supported.')