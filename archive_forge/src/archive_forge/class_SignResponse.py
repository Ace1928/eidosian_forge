import base64
import json
from pyu2f import errors
class SignResponse(object):

    def __init__(self, key_handle, signature_data, client_data):
        self.key_handle = key_handle
        self.signature_data = signature_data
        self.client_data = client_data