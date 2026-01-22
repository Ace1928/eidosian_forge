from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SslPoliciesListAvailableFeaturesResponse(_messages.Message):
    """A SslPoliciesListAvailableFeaturesResponse object.

  Fields:
    features: A string attribute.
  """
    features = _messages.StringField(1, repeated=True)