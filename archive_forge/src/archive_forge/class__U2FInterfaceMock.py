import base64
import json
import os
import unittest
import mock
from google_reauth import challenges, errors
import pyu2f
class _U2FInterfaceMock(object):

    def Authenticate(self, unused_app_id, challenge, unused_registered_keys):
        raise self.error