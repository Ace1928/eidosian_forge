import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
class AttributeValueWithCharsetSubstitution(str):
    """A stand-in object for a character encoding specified in HTML."""