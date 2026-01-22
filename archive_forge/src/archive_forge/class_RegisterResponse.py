import base64
import json
from pyu2f import errors
class RegisterResponse(object):

    def __init__(self, registration_data, client_data):
        self.registration_data = registration_data
        self.client_data = client_data