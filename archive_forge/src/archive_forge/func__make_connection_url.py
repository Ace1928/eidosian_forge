import copy
from http import client as http_client
import io
import logging
import os
import socket
import ssl
from urllib import parse as urlparse
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
from oslo_utils import importutils
from magnumclient import exceptions
def _make_connection_url(self, url):
    _class, _args, _kwargs = self.connection_params
    base_url = _args[2]
    return '%s/%s' % (base_url, url.lstrip('/'))