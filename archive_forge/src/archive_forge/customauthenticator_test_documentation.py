import base64
import json
import struct
import sys
import mock
from pyu2f import errors
from pyu2f import model
from pyu2f.convenience import customauthenticator
Test when plugin with error 'wrong data'.