from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Clustering(_messages.Message):
    """Configures table clustering.

  Fields:
    fields: One or more fields on which data should be clustered. Only top-
      level, non-repeated, simple-type fields are supported. The ordering of
      the clustering fields should be prioritized from most to least important
      for filtering purposes. Additional information on limitations can be
      found here: https://cloud.google.com/bigquery/docs/creating-clustered-
      tables#limitations
  """
    fields = _messages.StringField(1, repeated=True)