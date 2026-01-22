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
class FakeServiceCatalog(object):

    def url_for(self, endpoint_type, service_type, attr=None, filter_value=None):
        if attr == 'region' and filter_value:
            return 'http://regionhost:6385/v1/f14b41234'
        else:
            return 'http://localhost:6385/v1/f14b41234'