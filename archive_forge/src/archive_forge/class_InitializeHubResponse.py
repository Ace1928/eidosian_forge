from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InitializeHubResponse(_messages.Message):
    """Response message for the InitializeHub method.

  Fields:
    serviceIdentity: Name of the Hub default service identity, in the format:
      service-@gcp-sa-gkehub.iam.gserviceaccount.com The service account has
      `roles/gkehub.serviceAgent` in the Hub project.
    workloadIdentityPool: The Workload Identity Pool used for Workload
      Identity-enabled clusters registered with this Hub. Format:
      `.hub.id.goog`
  """
    serviceIdentity = _messages.StringField(1)
    workloadIdentityPool = _messages.StringField(2)