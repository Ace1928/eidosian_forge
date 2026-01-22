from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsGetRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsGetRequest
  object.

  Fields:
    name: Required. Resource name for the Authorized Orgs Desc. Format: `acces
      sPolicies/{policy_id}/authorizedOrgsDescs/{authorized_orgs_descs_id}`
  """
    name = _messages.StringField(1, required=True)