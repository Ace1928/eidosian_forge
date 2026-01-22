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
def GetInstanceScopes(self):
    relative_url = 'instance/service-accounts/{0}/scopes'.format(self.__service_account_name)
    response = _GceMetadataRequest(relative_url)
    return util.NormalizeScopes((six.ensure_str(scope).strip() for scope in response.readlines()))