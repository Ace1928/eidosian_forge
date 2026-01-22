from __future__ import unicode_literals
import re
from base64 import b64encode
import xml.etree.ElementTree as ET
import warnings
from winrm.protocol import Protocol
def _clean_error_msg(self, msg):
    """converts a Powershell CLIXML message to a more human readable string
        """
    if msg.startswith(b'#< CLIXML\r\n'):
        msg_xml = msg[11:]
        try:
            msg_xml = self._strip_namespace(msg_xml)
            root = ET.fromstring(msg_xml)
            nodes = root.findall('./S')
            new_msg = ''
            for s in nodes:
                new_msg += s.text.replace('_x000D__x000A_', '\n')
        except Exception as e:
            warnings.warn('There was a problem converting the Powershell error message: %s' % e)
        else:
            if len(new_msg):
                return new_msg.strip().encode('utf-8')
    return msg