import typing as T
import pytest
from docstring_parser.common import ParseError, RenderingStyle
from docstring_parser.epydoc import compose, parse
Test abbreviated docstring with only return type information.