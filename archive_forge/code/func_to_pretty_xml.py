from __future__ import annotations
import abc
import dataclasses
import datetime
import decimal
from xml.dom import minidom
from xml.etree import ElementTree as ET
def to_pretty_xml(self) -> str:
    """Return a pretty formatted XML string representing this instance."""
    return _pretty_xml(self.get_xml_element())