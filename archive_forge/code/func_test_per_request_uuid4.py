from functools import wraps
import os
import sys
import types
import uuid
from http.client import IncompleteRead
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
from cherrypy.test import helper
def test_per_request_uuid4(self):
    self.getPage('/request_uuid4')
    first_uuid4, _, second_uuid4 = self.body.decode().partition(' ')
    assert uuid.UUID(first_uuid4, version=4) == uuid.UUID(second_uuid4, version=4)
    self.getPage('/request_uuid4')
    third_uuid4, _, _ = self.body.decode().partition(' ')
    assert uuid.UUID(first_uuid4, version=4) != uuid.UUID(third_uuid4, version=4)