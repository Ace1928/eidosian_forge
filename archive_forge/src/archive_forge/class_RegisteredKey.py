import base64
import json
from pyu2f import errors
class RegisteredKey(object):

    def __init__(self, key_handle, version=u'U2F_V2'):
        self.key_handle = key_handle
        self.version = version