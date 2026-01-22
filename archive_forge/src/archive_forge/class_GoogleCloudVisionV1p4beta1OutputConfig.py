from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p4beta1OutputConfig(_messages.Message):
    """The desired output location and metadata.

  Fields:
    batchSize: The max number of response protos to put into each output JSON
      file on Google Cloud Storage. The valid range is [1, 100]. If not
      specified, the default value is 20. For example, for one pdf file with
      100 pages, 100 response protos will be generated. If `batch_size` = 20,
      then 5 json files each containing 20 response protos will be written
      under the prefix `gcs_destination`.`uri`. Currently, batch_size only
      applies to GcsDestination, with potential future support for other
      output configurations.
    gcsDestination: The Google Cloud Storage location to write the output(s)
      to.
  """
    batchSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    gcsDestination = _messages.MessageField('GoogleCloudVisionV1p4beta1GcsDestination', 2)