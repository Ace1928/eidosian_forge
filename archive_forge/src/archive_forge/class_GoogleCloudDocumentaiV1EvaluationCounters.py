from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1EvaluationCounters(_messages.Message):
    """Evaluation counters for the documents that were used.

  Fields:
    evaluatedDocumentsCount: How many documents were used in the evaluation.
    failedDocumentsCount: How many documents were not included in the
      evaluation as Document AI failed to process them.
    inputDocumentsCount: How many documents were sent for evaluation.
    invalidDocumentsCount: How many documents were not included in the
      evaluation as they didn't pass validation.
  """
    evaluatedDocumentsCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    failedDocumentsCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    inputDocumentsCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    invalidDocumentsCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)