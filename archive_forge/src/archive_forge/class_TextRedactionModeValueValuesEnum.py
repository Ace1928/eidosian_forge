from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TextRedactionModeValueValuesEnum(_messages.Enum):
    """Determines how to redact text from images.

    Values:
      TEXT_REDACTION_MODE_UNSPECIFIED: No text redaction specified. Same as
        REDACT_NO_TEXT.
      REDACT_ALL_TEXT: Redact all text.
      REDACT_SENSITIVE_TEXT: Redact sensitive text. Uses the set of [Default
        DICOM InfoTypes](https://cloud.google.com/healthcare-api/docs/how-
        tos/dicom-deidentify#default_dicom_infotypes).
      REDACT_NO_TEXT: Do not redact text.
      REDACT_SENSITIVE_TEXT_CLEAN_DESCRIPTORS: This mode is like
        `REDACT_SENSITIVE_TEXT` with the addition of the [Clean Descriptors
        Option] (https://dicom.nema.org/medical/dicom/2018e/output/chtml/part1
        5/sect_E.3.5.html) enabled: When cleaning text, the process attempts
        to transform phrases matching any of the tags marked for removal
        (action codes D, Z, X, and U) in the [Basic Profile] (https://dicom.ne
        ma.org/medical/dicom/2018e/output/chtml/part15/chapter_E.html). These
        contextual phrases are replaced with the token "[CTX]". This mode uses
        an additional infoType during inspection.
    """
    TEXT_REDACTION_MODE_UNSPECIFIED = 0
    REDACT_ALL_TEXT = 1
    REDACT_SENSITIVE_TEXT = 2
    REDACT_NO_TEXT = 3
    REDACT_SENSITIVE_TEXT_CLEAN_DESCRIPTORS = 4