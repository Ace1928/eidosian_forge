from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamAdminV1WorkforcePoolProviderExtraAttributesOAuth2ClientQueryParameters(_messages.Message):
    """Represents the parameters to control which claims are fetched from an
  IdP.

  Fields:
    filter: Optional. The filter used to request specific records from IdP. In
      case of attributes type as AZURE_AD_GROUPS_MAIL, it represents the
      filter used to request specific groups for users from IdP. By default
      all the groups associated with the user are fetched. The groups that are
      used should be mail enabled and security enabled. See
      https://learn.microsoft.com/en-us/graph/search-query-parameter for more
      details.
  """
    filter = _messages.StringField(1)