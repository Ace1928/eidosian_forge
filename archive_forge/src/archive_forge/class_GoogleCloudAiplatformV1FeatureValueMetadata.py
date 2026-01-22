from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FeatureValueMetadata(_messages.Message):
    """Metadata of feature value.

  Fields:
    generateTime: Feature generation timestamp. Typically, it is provided by
      user at feature ingestion time. If not, feature store will use the
      system timestamp when the data is ingested into feature store. For
      streaming ingestion, the time, aligned by days, must be no older than
      five years (1825 days) and no later than one year (366 days) in the
      future.
  """
    generateTime = _messages.StringField(1)