from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FilterSplit(_messages.Message):
    """Assigns input data to training, validation, and test sets based on the
  given filters, data pieces not matched by any filter are ignored. Currently
  only supported for Datasets containing DataItems. If any of the filters in
  this message are to match nothing, then they can be set as '-' (the minus
  sign). Supported only for unstructured Datasets.

  Fields:
    testFilter: Required. A filter on DataItems of the Dataset. DataItems that
      match this filter are used to test the Model. A filter with same syntax
      as the one used in DatasetService.ListDataItems may be used. If a single
      DataItem is matched by more than one of the FilterSplit filters, then it
      is assigned to the first set that applies to it in the training,
      validation, test order.
    trainingFilter: Required. A filter on DataItems of the Dataset. DataItems
      that match this filter are used to train the Model. A filter with same
      syntax as the one used in DatasetService.ListDataItems may be used. If a
      single DataItem is matched by more than one of the FilterSplit filters,
      then it is assigned to the first set that applies to it in the training,
      validation, test order.
    validationFilter: Required. A filter on DataItems of the Dataset.
      DataItems that match this filter are used to validate the Model. A
      filter with same syntax as the one used in DatasetService.ListDataItems
      may be used. If a single DataItem is matched by more than one of the
      FilterSplit filters, then it is assigned to the first set that applies
      to it in the training, validation, test order.
  """
    testFilter = _messages.StringField(1)
    trainingFilter = _messages.StringField(2)
    validationFilter = _messages.StringField(3)