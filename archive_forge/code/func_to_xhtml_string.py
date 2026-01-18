from __future__ import annotations
from xml.etree.ElementTree import ProcessingInstruction
from xml.etree.ElementTree import Comment, ElementTree, Element, QName, HTML_EMPTY
import re
from typing import Callable, Literal, NoReturn
def to_xhtml_string(element: Element) -> str:
    """ Serialize element and its children to a string of XHTML. """
    return _write_html(ElementTree(element).getroot(), format='xhtml')