from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesTablesAuthorizedViewsCreateRequest(_messages.Message):
    """A BigtableadminProjectsInstancesTablesAuthorizedViewsCreateRequest
  object.

  Fields:
    authorizedView: A AuthorizedView resource to be passed as the request
      body.
    authorizedViewId: Required. The id of the AuthorizedView to create. This
      AuthorizedView must not already exist. The `authorized_view_id` appended
      to `parent` forms the full AuthorizedView name of the form `projects/{pr
      oject}/instances/{instance}/tables/{table}/authorizedView/{authorized_vi
      ew}`.
    parent: Required. This is the name of the table the AuthorizedView belongs
      to. Values are of the form
      `projects/{project}/instances/{instance}/tables/{table}`.
  """
    authorizedView = _messages.MessageField('AuthorizedView', 1)
    authorizedViewId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)