import requests
import re
import struct
import sys
from winrm.exceptions import WinRMError
def _decrypt_ntlm_message(self, encrypted_data, host):
    signature_length = struct.unpack('<i', encrypted_data[:4])[0]
    signature = encrypted_data[4:signature_length + 4]
    encrypted_message = encrypted_data[signature_length + 4:]
    message = self.session.auth.session_security.unwrap(encrypted_message, signature)
    return message