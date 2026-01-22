from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1StratifiedSplit(_messages.Message):
    """Assigns input data to the training, validation, and test sets so that
  the distribution of values found in the categorical column (as specified by
  the `key` field) is mirrored within each split. The fraction values
  determine the relative sizes of the splits. For example, if the specified
  column has three values, with 50% of the rows having value "A", 25% value
  "B", and 25% value "C", and the split fractions are specified as 80/10/10,
  then the training set will constitute 80% of the training data, with about
  50% of the training set rows having the value "A" for the specified column,
  about 25% having the value "B", and about 25% having the value "C". Only the
  top 500 occurring values are used; any values not in the top 500 values are
  randomly assigned to a split. If less than three rows contain a specific
  value, those rows are randomly assigned. Supported only for tabular
  Datasets.

  Fields:
    key: Required. The key is a name of one of the Dataset's data columns. The
      key provided must be for a categorical column.
    testFraction: The fraction of the input data that is to be used to
      evaluate the Model.
    trainingFraction: The fraction of the input data that is to be used to
      train the Model.
    validationFraction: The fraction of the input data that is to be used to
      validate the Model.
  """
    key = _messages.StringField(1)
    testFraction = _messages.FloatField(2)
    trainingFraction = _messages.FloatField(3)
    validationFraction = _messages.FloatField(4)