from __future__ import unicode_literals
import sys
import os
import requests
import requests.auth
import warnings
from winrm.exceptions import InvalidCredentialsError, WinRMError, WinRMTransportError
from winrm.encryption import Encryption
class UnsupportedAuthArgument(Warning):
    pass