from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.util import text
def AddIndexConfigToUpdateRequest(unused_ref, args, req):
    """Update patch request to include indexConfig.

  The mapping of index config message to API behavior is as follows:
    None          - Clears the exemption
    indexes=[]    - Disables all indexes
    indexes=[...] - Sets the index config to the indexes provided

  Args:
    unused_ref: The field resource reference.
    args: The parsed arg namespace.
    req: The auto-generated patch request.
  Returns:
    FirestoreProjectsDatabasesCollectionGroupsFieldsPatchRequest
  """
    messages = GetMessagesModule()
    if args.disable_indexes:
        index_config = messages.GoogleFirestoreAdminV1IndexConfig(indexes=[])
    elif args.IsSpecified('index') and req.googleFirestoreAdminV1Field:
        index_config = req.googleFirestoreAdminV1Field.indexConfig
    else:
        index_config = None
    arg_utils.SetFieldInMessage(req, 'googleFirestoreAdminV1Field.indexConfig', index_config)
    return req