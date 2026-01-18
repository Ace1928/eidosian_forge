import json
import time
from unittest import mock
from zaqarclient.queues.v1 import claim
from zaqarclient.tests.queues import base
from zaqarclient.transport import errors
from zaqarclient.transport import response
def verify_limit(request):
    self.assertIn('limit', request.params)
    self.assertEqual(10, request.params['limit'])
    return response.Response(None, "{0: [], 'messages': []}")