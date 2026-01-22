import abc
import copy
import functools
import urllib
import warnings
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import plugin
from oslo_utils import strutils
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.i18n import _
def _prepare_return_value(self, http_response, data):
    if self.client.include_metadata:
        return Response(http_response, data)
    return data