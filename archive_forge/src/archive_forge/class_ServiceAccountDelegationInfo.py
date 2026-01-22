from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceAccountDelegationInfo(_messages.Message):
    """Identity delegation history of an authenticated service account.

  Fields:
    principalEmail: The email address of a Google account.
    principalSubject: A string representing the principal_subject associated
      with the identity. As compared to `principal_email`, supports principals
      that aren't associated with email addresses, such as third party
      principals. For most identities, the format will be
      `principal://iam.googleapis.com/{identity pool name}/subjects/{subject}`
      except for some GKE identities (GKE_WORKLOAD, FREEFORM,
      GKE_HUB_WORKLOAD) that are still in the legacy format
      `serviceAccount:{identity pool name}[{subject}]`
  """
    principalEmail = _messages.StringField(1)
    principalSubject = _messages.StringField(2)