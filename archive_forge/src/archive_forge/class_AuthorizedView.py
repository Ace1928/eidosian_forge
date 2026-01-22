from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthorizedView(_messages.Message):
    """An Authorized View of a Cloud Bigtable Table.

  Fields:
    deletionProtection: Set to true to make the AuthorizedView protected
      against deletion. The parent Table and containing Instance cannot be
      deleted if an AuthorizedView has this bit set.
    etag: The etag for this AuthorizedView. If this is provided on update, it
      must match the server's etag. The server returns ABORTED error on a
      mismatched etag.
    name: Identifier. The name of this AuthorizedView. Values are of the form
      `projects/{project}/instances/{instance}/tables/{table}/authorizedViews/
      {authorized_view}`
    subsetView: An AuthorizedView permitting access to an explicit subset of a
      Table.
  """
    deletionProtection = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    name = _messages.StringField(3)
    subsetView = _messages.MessageField('GoogleBigtableAdminV2AuthorizedViewSubsetView', 4)