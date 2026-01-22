from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class APIDef(object):
    """Struct for info required to instantiate clients/messages for API versions.

  Attributes:
    apitools: ApitoolsClientDef for this API version.
    gapic: GapicClientDef for this API version.
    default_version: bool, Whether this API version is the default version for
      the API.
    enable_mtls: bool, Whether this API version supports mTLS.
    mtls_endpoint_override: str, The mTLS endpoint for this API version. If
      empty, the MTLS_BASE_URL in the API client will be used.
  """

    def __init__(self, apitools=None, gapic=None, default_version=False, enable_mtls=False, mtls_endpoint_override=''):
        self.apitools = apitools
        self.gapic = gapic
        self.default_version = default_version
        self.enable_mtls = enable_mtls
        self.mtls_endpoint_override = mtls_endpoint_override

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_init_source(self):
        src_fmt = 'APIDef({0}, {1}, {2}, {3}, "{4}")'
        return src_fmt.format(self.apitools, self.gapic, self.default_version, self.enable_mtls, self.mtls_endpoint_override)

    def __repr__(self):
        return self.get_init_source()