from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataSplitResult(_messages.Message):
    """Data split result. This contains references to the training and
  evaluation data tables that were used to train the model.

  Fields:
    evaluationTable: Table reference of the evaluation data after split.
    testTable: Table reference of the test data after split.
    trainingTable: Table reference of the training data after split.
  """
    evaluationTable = _messages.MessageField('TableReference', 1)
    testTable = _messages.MessageField('TableReference', 2)
    trainingTable = _messages.MessageField('TableReference', 3)