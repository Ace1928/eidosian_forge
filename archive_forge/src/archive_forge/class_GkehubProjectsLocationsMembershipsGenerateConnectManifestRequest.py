from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsMembershipsGenerateConnectManifestRequest(_messages.Message):
    """A GkehubProjectsLocationsMembershipsGenerateConnectManifestRequest
  object.

  Fields:
    cpuRequest: Optional. To customize the CPU requested for the Connect Agent
      resource.
    imagePullSecretContent: Optional. The image pull secret content for the
      registry, if not public.
    isUpgrade: Optional. If true, generate the resources for upgrade only.
      Some resources generated only for installation (e.g. secrets) will be
      excluded.
    memLimit: Optional. To customize the memory limit for the Connect Agent
      resource.
    memRequest: Optional. To customize the memory requested for the Connect
      Agent resource.
    name: Required. The Membership resource name the Agent will associate
      with, in the format `projects/*/locations/*/memberships/*`.
    namespace: Optional. Namespace for GKE Connect agent resources. Defaults
      to `gke-connect`. The Connect Agent is authorized automatically when run
      in the default namespace. Otherwise, explicit authorization must be
      granted with an additional IAM binding.
    proxy: Optional. URI of a proxy if connectivity from the agent to
      gkeconnect.googleapis.com requires the use of a proxy. Format must be in
      the form `http(s)://{proxy_address}`, depending on the HTTP/HTTPS
      protocol supported by the proxy. This will direct the connect agent's
      outbound traffic through a HTTP(S) proxy.
    registry: Optional. The registry to fetch the connect agent image from.
      Defaults to gcr.io/gkeconnect.
    version: Optional. The Connect agent version to use. Defaults to the most
      current version.
  """
    cpuRequest = _messages.StringField(1)
    imagePullSecretContent = _messages.BytesField(2)
    isUpgrade = _messages.BooleanField(3)
    memLimit = _messages.StringField(4)
    memRequest = _messages.StringField(5)
    name = _messages.StringField(6, required=True)
    namespace = _messages.StringField(7)
    proxy = _messages.BytesField(8)
    registry = _messages.StringField(9)
    version = _messages.StringField(10)