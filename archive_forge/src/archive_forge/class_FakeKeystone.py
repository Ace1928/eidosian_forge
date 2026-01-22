import copy
import datetime
import io
import os
from oslo_serialization import jsonutils
import queue
import sys
import fixtures
import testtools
from magnumclient.common import httpclient as http
from magnumclient import shell
class FakeKeystone(object):
    service_catalog = FakeServiceCatalog()
    timestamp = datetime.datetime.utcnow() + datetime.timedelta(days=5)

    def __init__(self, auth_token):
        self.auth_token = auth_token
        self.auth_ref = {'token': {'expires': FakeKeystone.timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f'), 'id': 'd1a541311782870742235'}}