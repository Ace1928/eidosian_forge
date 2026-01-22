from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Protocol
from cssselect import GenericTranslator as OriginalGenericTranslator
from cssselect import HTMLTranslator as OriginalHTMLTranslator
from cssselect.parser import Element, FunctionalPseudoElement, PseudoElement
from cssselect.xpath import ExpressionError
from cssselect.xpath import XPathExpr as OriginalXPathExpr
class TranslatorMixin:
    """This mixin adds support to CSS pseudo elements via dynamic dispatch.

    Currently supported pseudo-elements are ``::text`` and ``::attr(ATTR_NAME)``.
    """

    def xpath_element(self: TranslatorProtocol, selector: Element) -> XPathExpr:
        xpath = super().xpath_element(selector)
        return XPathExpr.from_xpath(xpath)

    def xpath_pseudo_element(self, xpath: OriginalXPathExpr, pseudo_element: PseudoElement) -> OriginalXPathExpr:
        """
        Dispatch method that transforms XPath to support pseudo-element
        """
        if isinstance(pseudo_element, FunctionalPseudoElement):
            method_name = f'xpath_{pseudo_element.name.replace('-', '_')}_functional_pseudo_element'
            method = getattr(self, method_name, None)
            if not method:
                raise ExpressionError(f'The functional pseudo-element ::{pseudo_element.name}() is unknown')
            xpath = method(xpath, pseudo_element)
        else:
            method_name = f'xpath_{pseudo_element.replace('-', '_')}_simple_pseudo_element'
            method = getattr(self, method_name, None)
            if not method:
                raise ExpressionError(f'The pseudo-element ::{pseudo_element} is unknown')
            xpath = method(xpath)
        return xpath

    def xpath_attr_functional_pseudo_element(self, xpath: OriginalXPathExpr, function: FunctionalPseudoElement) -> XPathExpr:
        """Support selecting attribute values using ::attr() pseudo-element"""
        if function.argument_types() not in (['STRING'], ['IDENT']):
            raise ExpressionError(f'Expected a single string or ident for ::attr(), got {function.arguments!r}')
        return XPathExpr.from_xpath(xpath, attribute=function.arguments[0].value)

    def xpath_text_simple_pseudo_element(self, xpath: OriginalXPathExpr) -> XPathExpr:
        """Support selecting text nodes using ::text pseudo-element"""
        return XPathExpr.from_xpath(xpath, textnode=True)