import railroad
from pip._vendor import pyparsing
import typing
from typing import (
from jinja2 import Template
from io import StringIO
import inspect
def _worth_extracting(element: pyparsing.ParserElement) -> bool:
    """
    Returns true if this element is worth having its own sub-diagram. Simply, if any of its children
    themselves have children, then its complex enough to extract
    """
    children = element.recurse()
    return any((child.recurse() for child in children))