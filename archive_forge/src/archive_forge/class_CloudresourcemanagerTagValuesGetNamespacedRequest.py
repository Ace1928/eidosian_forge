from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerTagValuesGetNamespacedRequest(_messages.Message):
    """A CloudresourcemanagerTagValuesGetNamespacedRequest object.

  Fields:
    name: Required. A namespaced tag value name in the following format:
      `{parentId}/{tagKeyShort}/{tagValueShort}` Examples: - `42/foo/abc` for
      a value with short name "abc" under the key with short name "foo" under
      the organization with ID 42 - `r2-d2/bar/xyz` for a value with short
      name "xyz" under the key with short name "bar" under the project with ID
      "r2-d2"
  """
    name = _messages.StringField(1)