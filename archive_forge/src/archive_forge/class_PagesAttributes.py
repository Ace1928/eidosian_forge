from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class PagesAttributes:
    """Page Attributes, Table 6.2, Page 52."""
    TYPE = '/Type'
    KIDS = '/Kids'
    COUNT = '/Count'
    PARENT = '/Parent'