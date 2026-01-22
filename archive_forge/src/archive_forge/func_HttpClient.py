from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import base64
import contextlib
import os
import re
import ssl
import sys
import tempfile
from googlecloudsdk.api_lib.run import gke
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.util import files
import requests
import six
from six.moves.urllib import parse as urlparse
def HttpClient(self):
    assert self.active
    if not self.client_key and self.client_cert and self.client_cert_domain:
        raise ValueError('Kubeconfig authentication requires a client certificate authentication method.')
    if self.client_cert_domain:
        from googlecloudsdk.core import transports
        http_client = transports.GetApitoolsTransport(response_encoding=transport.ENCODING, ca_certs=self.ca_certs, client_certificate=self.client_cert, client_key=self.client_key, client_cert_domain=self.client_cert_domain)
        return http_client
    from googlecloudsdk.core.credentials import transports
    http_client = transports.GetApitoolsTransport(response_encoding=transport.ENCODING, ca_certs=self.ca_certs)
    return http_client