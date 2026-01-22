from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerTagKeysGetNamespacedRequest(_messages.Message):
    """A CloudresourcemanagerTagKeysGetNamespacedRequest object.

  Fields:
    name: Required. A namespaced tag key name in the format
      `{parentId}/{tagKeyShort}`, such as `42/foo` for a key with short name
      "foo" under the organization with ID 42 or `r2-d2/bar` for a key with
      short name "bar" under the project `r2-d2`.
  """
    name = _messages.StringField(1)