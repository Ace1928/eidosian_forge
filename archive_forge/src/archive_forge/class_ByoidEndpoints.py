from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import json
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
class ByoidEndpoints(object):
    """Base class for BYOID endpoints.
  """

    def __init__(self, service, enable_mtls=False, universe_domain='googleapis.com'):
        self._sts_template = 'https://{service}.{mtls}{universe}'
        self._service = service
        self._mtls = 'mtls.' if enable_mtls else ''
        self._universe_domain = universe_domain

    @property
    def _base_url(self):
        return self._sts_template.format(service=self._service, mtls=self._mtls, universe=self._universe_domain)