from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RollbackFhirResourceFilteringFields(_messages.Message):
    """Filters to select resources that need to be rolled back.

  Fields:
    metadataFilter: Optional. A filter expression that matches data in the
      `Resource.meta` element. Supports all filters in
      [AIP-160](https://google.aip.dev/160) except the "has" (`:`) operator.
      Supports the following custom functions: * `tag("") = ""` for tag
      filtering. * `extension_value_ts("") = ` for filtering extensions with a
      timestamp, where `` is a Unix timestamp. Supports the `>`, `<`, `<=`,
      `>=`, and `!=` comparison operators.
    operationIds: Optional. A list of operation IDs to roll back. Only changes
      made by these operations will be rolled back.
  """
    metadataFilter = _messages.StringField(1)
    operationIds = _messages.IntegerField(2, repeated=True, variant=_messages.Variant.UINT64)