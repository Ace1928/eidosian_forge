from __future__ import (absolute_import, division, print_function)
import base64
import os
import re
import shlex
import pkgutil
import xml.etree.ElementTree as ET
import ntpath
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.plugins.shell import ShellBase
def _parse_clixml(data, stream='Error'):
    """
    Takes a byte string like '#< CLIXML\r
<Objs...' and extracts the stream
    message encoded in the XML data. CLIXML is used by PowerShell to encode
    multiple objects in stderr.
    """
    lines = []
    while data:
        end_idx = data.find(b'</Objs>') + 7
        current_element = data[data.find(b'<Objs '):end_idx]
        data = data[end_idx:]
        clixml = ET.fromstring(current_element)
        namespace_match = re.match('{(.*)}', clixml.tag)
        namespace = '{%s}' % namespace_match.group(1) if namespace_match else ''
        strings = clixml.findall('./%sS' % namespace)
        lines.extend([e.text.replace('_x000D__x000A_', '') for e in strings if e.attrib.get('S') == stream])
    return to_bytes('\r\n'.join(lines))