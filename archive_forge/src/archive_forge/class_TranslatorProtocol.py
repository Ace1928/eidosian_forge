from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Protocol
from cssselect import GenericTranslator as OriginalGenericTranslator
from cssselect import HTMLTranslator as OriginalHTMLTranslator
from cssselect.parser import Element, FunctionalPseudoElement, PseudoElement
from cssselect.xpath import ExpressionError
from cssselect.xpath import XPathExpr as OriginalXPathExpr
class TranslatorProtocol(Protocol):

    def xpath_element(self, selector: Element) -> OriginalXPathExpr:
        pass

    def css_to_xpath(self, css: str, prefix: str=...) -> str:
        pass