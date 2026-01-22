from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class UnderscoreProcessor(AsteriskProcessor):
    """Emphasis processor for handling strong and em matches inside underscores."""
    PATTERNS = [EmStrongItem(re.compile(EM_STRONG2_RE, re.DOTALL | re.UNICODE), 'double', 'strong,em'), EmStrongItem(re.compile(STRONG_EM2_RE, re.DOTALL | re.UNICODE), 'double', 'em,strong'), EmStrongItem(re.compile(SMART_STRONG_EM_RE, re.DOTALL | re.UNICODE), 'double2', 'strong,em'), EmStrongItem(re.compile(SMART_STRONG_RE, re.DOTALL | re.UNICODE), 'single', 'strong'), EmStrongItem(re.compile(SMART_EMPHASIS_RE, re.DOTALL | re.UNICODE), 'single', 'em')]
    ' The various strong and emphasis patterns handled by this processor. '