from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1CrossRegionalSource(_messages.Message):
    """Cross-regional source used to import an existing taxonomy into a
  different region.

  Fields:
    taxonomy: Required. The resource name of the source taxonomy to import.
  """
    taxonomy = _messages.StringField(1)