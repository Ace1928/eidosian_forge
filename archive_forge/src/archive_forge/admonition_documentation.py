from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
import xml.etree.ElementTree as etree
import re
from typing import TYPE_CHECKING
Get sibling admonition.

        Retrieve the appropriate sibling element. This can get tricky when
        dealing with lists.

        