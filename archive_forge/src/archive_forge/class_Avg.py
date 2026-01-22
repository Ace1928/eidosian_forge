from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Avg(_messages.Message):
    """Average of the values of the requested field. * Only numeric values will
  be aggregated. All non-numeric values including `NULL` are skipped. * If the
  aggregated values contain `NaN`, returns `NaN`. Infinity math follows
  IEEE-754 standards. * If the aggregated value set is empty, returns `NULL`.
  * Always returns the result as a double.

  Fields:
    field: The field to aggregate on.
  """
    field = _messages.MessageField('FieldReference', 1)