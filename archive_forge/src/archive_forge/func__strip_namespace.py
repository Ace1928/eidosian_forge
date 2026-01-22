from __future__ import unicode_literals
import re
from base64 import b64encode
import xml.etree.ElementTree as ET
import warnings
from winrm.protocol import Protocol
def _strip_namespace(self, xml):
    """strips any namespaces from an xml string"""
    p = re.compile(b'xmlns=*[""][^""]*[""]')
    allmatches = p.finditer(xml)
    for match in allmatches:
        xml = xml.replace(match.group(), b'')
    return xml