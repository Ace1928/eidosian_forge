import requests
import re
import struct
import sys
from winrm.exceptions import WinRMError
def _encrypt_message(self, message, host):
    message_length = str(len(message)).encode()
    encrypted_stream = self._build_message(message, host)
    message_payload = self.MIME_BOUNDARY + b'\r\n\tContent-Type: ' + self.protocol_string + b'\r\n\tOriginalContent: type=application/soap+xml;charset=UTF-8;Length=' + message_length + b'\r\n' + self.MIME_BOUNDARY + b'\r\n\tContent-Type: application/octet-stream\r\n' + encrypted_stream
    return message_payload