from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Protocol
from cssselect import GenericTranslator as OriginalGenericTranslator
from cssselect import HTMLTranslator as OriginalHTMLTranslator
from cssselect.parser import Element, FunctionalPseudoElement, PseudoElement
from cssselect.xpath import ExpressionError
from cssselect.xpath import XPathExpr as OriginalXPathExpr
def xpath_text_simple_pseudo_element(self, xpath: OriginalXPathExpr) -> XPathExpr:
    """Support selecting text nodes using ::text pseudo-element"""
    return XPathExpr.from_xpath(xpath, textnode=True)