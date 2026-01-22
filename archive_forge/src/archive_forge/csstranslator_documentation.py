from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Protocol
from cssselect import GenericTranslator as OriginalGenericTranslator
from cssselect import HTMLTranslator as OriginalHTMLTranslator
from cssselect.parser import Element, FunctionalPseudoElement, PseudoElement
from cssselect.xpath import ExpressionError
from cssselect.xpath import XPathExpr as OriginalXPathExpr
Support selecting text nodes using ::text pseudo-element