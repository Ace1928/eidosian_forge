from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentCorrectnessValueValuesEnum(_messages.Enum):
    """Optional. Whether or not the information in the document is correct.
    For example: * Query: "Can I return the package in 2 days once received?"
    * Suggested document says: "Items must be returned/exchanged within 60
    days of the purchase date." * Ground truth: "No return or exchange is
    allowed." * [document_correctness]: INCORRECT

    Values:
      DOCUMENT_CORRECTNESS_UNSPECIFIED: Document correctness unspecified.
      INCORRECT: Information in document is incorrect.
      CORRECT: Information in document is correct.
    """
    DOCUMENT_CORRECTNESS_UNSPECIFIED = 0
    INCORRECT = 1
    CORRECT = 2