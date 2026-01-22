import base64
import hashlib
import json
import os
import struct
import subprocess
import sys
from pyu2f import errors
from pyu2f import model
from pyu2f.convenience import baseauthenticator
def _Base64Encode(self, bytes_data):
    """Helper method to base64 encode, strip padding, and return str
      result."""
    return base64.urlsafe_b64encode(bytes_data).decode().rstrip('=')