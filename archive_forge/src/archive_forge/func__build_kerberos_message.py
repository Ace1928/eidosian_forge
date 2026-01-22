import requests
import re
import struct
import sys
from winrm.exceptions import WinRMError
def _build_kerberos_message(self, message, host):
    sealed_message, signature = self.session.auth.wrap_winrm(host, message)
    signature_length = struct.pack('<i', len(signature))
    return signature_length + signature + sealed_message