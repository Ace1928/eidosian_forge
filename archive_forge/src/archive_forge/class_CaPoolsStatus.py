from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CaPoolsStatus(_messages.Message):
    """Status of CA pools in a region.

  Fields:
    caPools: The CA pool string has a relative resource path following the
      form "projects/{project number}/locations/{location}/caPools/{CA pool}".
  """
    caPools = _messages.StringField(1, repeated=True)