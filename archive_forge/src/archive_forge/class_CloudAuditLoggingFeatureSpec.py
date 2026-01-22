from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAuditLoggingFeatureSpec(_messages.Message):
    """**Cloud Audit Logging**: Spec for Audit Logging Allowlisting.

  Fields:
    allowlistedServiceAccounts: Service account that should be allowlisted to
      send the audit logs; eg cloudauditlogging@gcp-
      project.iam.gserviceaccount.com. These accounts must already exist, but
      do not need to have any permissions granted to them. The customer's
      entitlements will be checked prior to allowlisting (i.e. the customer
      must be an Anthos customer.)
  """
    allowlistedServiceAccounts = _messages.StringField(1, repeated=True)