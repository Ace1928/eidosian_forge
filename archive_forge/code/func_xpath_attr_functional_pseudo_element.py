from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Protocol
from cssselect import GenericTranslator as OriginalGenericTranslator
from cssselect import HTMLTranslator as OriginalHTMLTranslator
from cssselect.parser import Element, FunctionalPseudoElement, PseudoElement
from cssselect.xpath import ExpressionError
from cssselect.xpath import XPathExpr as OriginalXPathExpr
def xpath_attr_functional_pseudo_element(self, xpath: OriginalXPathExpr, function: FunctionalPseudoElement) -> XPathExpr:
    """Support selecting attribute values using ::attr() pseudo-element"""
    if function.argument_types() not in (['STRING'], ['IDENT']):
        raise ExpressionError(f'Expected a single string or ident for ::attr(), got {function.arguments!r}')
    return XPathExpr.from_xpath(xpath, attribute=function.arguments[0].value)