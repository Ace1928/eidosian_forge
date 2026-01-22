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
class GKEConnectionContext(ConnectionInfo):
    """Context manager to connect to the GKE Cloud Run add-in."""

    def __init__(self, cluster_ref, api_name, version):
        super(GKEConnectionContext, self).__init__(api_name, version)
        self.cluster_ref = cluster_ref

    @contextlib.contextmanager
    def Connect(self):
        _CheckTLSSupport()
        with gke.ClusterConnectionInfo(self.cluster_ref) as (ip, ca_certs):
            self.ca_certs = ca_certs
            with gke.MonkeypatchAddressChecking('kubernetes.default', ip) as endpoint:
                self.endpoint = 'https://{}/'.format(endpoint)
                with _OverrideEndpointOverrides(self._api_name, self.endpoint):
                    yield self

    @property
    def operator(self):
        return 'Cloud Run for Anthos'

    def HttpClient(self):
        assert self.active
        from googlecloudsdk.core.credentials import transports
        http_client = transports.GetApitoolsTransport(response_encoding=transport.ENCODING, ca_certs=self.ca_certs)
        return http_client

    @property
    def location_label(self):
        return ' of cluster [{{{{bold}}}}{}{{{{reset}}}}]'.format(self.cluster_name)

    @property
    def cluster_name(self):
        return self.cluster_ref.Name()

    @property
    def cluster_location(self):
        return self.cluster_ref.zone

    @property
    def cluster_project(self):
        return self.cluster_ref.projectId

    @property
    def supports_one_platform(self):
        return False

    @property
    def ns_label(self):
        return 'namespace'