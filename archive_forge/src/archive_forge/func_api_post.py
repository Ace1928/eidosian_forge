import abc
import atexit
import datetime
import errno
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from testtools import content as ttc
import textwrap
import time
from unittest import mock
import urllib.parse as urlparse
import uuid
import fixtures
import glance_store
from os_win import utilsfactory as os_win_utilsfactory
from oslo_config import cfg
from oslo_serialization import jsonutils
import testtools
import webob
from glance.common import config
from glance.common import utils
from glance.common import wsgi
from glance.db.sqlalchemy import api as db_api
from glance import tests as glance_tests
from glance.tests import utils as test_utils
import glance.async_
def api_post(self, url, headers=None, data=None, json=None, body_file=None):
    """Perform a POST request against the API.

        :param url: The *path* part of the URL to call (i.e. /v2/images)
        :param headers: Optional updates to the default set of headers
        :param data: Optional bytes data payload to send (overrides @json)
        :param json: Optional dict structure to be jsonified and sent as
                     the payload (mutually exclusive with @data)
        :param body_file: Optional io.IOBase to provide as the input data
                          stream for the request (overrides @data)
        :returns: A webob.Response object
        """
    return self.api_request('POST', url, headers=headers, data=data, json=json, body_file=body_file)