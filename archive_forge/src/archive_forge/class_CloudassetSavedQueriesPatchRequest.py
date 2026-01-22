from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetSavedQueriesPatchRequest(_messages.Message):
    """A CloudassetSavedQueriesPatchRequest object.

  Fields:
    name: The resource name of the saved query. The format must be: *
      projects/project_number/savedQueries/saved_query_id *
      folders/folder_number/savedQueries/saved_query_id *
      organizations/organization_number/savedQueries/saved_query_id
    savedQuery: A SavedQuery resource to be passed as the request body.
    updateMask: Required. The list of fields to update.
  """
    name = _messages.StringField(1, required=True)
    savedQuery = _messages.MessageField('SavedQuery', 2)
    updateMask = _messages.StringField(3)