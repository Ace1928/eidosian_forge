import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_pseudo(self, pseudo: Pseudo) -> XPathExpr:
    """Translate a pseudo-class."""
    method_name = 'xpath_%s_pseudo' % pseudo.ident.replace('-', '_')
    method = getattr(self, method_name, None)
    if not method:
        raise ExpressionError('The pseudo-class :%s is unknown' % pseudo.ident)
    return typing.cast(XPathExpr, method(self.xpath(pseudo.selector)))