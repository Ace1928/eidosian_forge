from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportProductSetsResponse(_messages.Message):
    """Response message for the `ImportProductSets` method. This message is
  returned by the google.longrunning.Operations.GetOperation method in the
  returned google.longrunning.Operation.response field.

  Fields:
    referenceImages: The list of reference_images that are imported
      successfully.
    statuses: The rpc status for each ImportProductSet request, including both
      successes and errors. The number of statuses here matches the number of
      lines in the csv file, and statuses[i] stores the success or failure
      status of processing the i-th line of the csv, starting from line 0.
  """
    referenceImages = _messages.MessageField('ReferenceImage', 1, repeated=True)
    statuses = _messages.MessageField('Status', 2, repeated=True)