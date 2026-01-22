from __future__ import print_function
import argparse
import contextlib
import datetime
import json
import os
import threading
import warnings
import httplib2
import oauth2client
import oauth2client.client
from oauth2client import service_account
from oauth2client import tools  # for gflags declarations
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.py import exceptions
from apitools.base.py import util
def GetServiceAccount(self, account):
    relative_url = 'instance/service-accounts'
    response = _GceMetadataRequest(relative_url)
    response_lines = [six.ensure_str(line).rstrip(u'/\n\r') for line in response.readlines()]
    return account in response_lines