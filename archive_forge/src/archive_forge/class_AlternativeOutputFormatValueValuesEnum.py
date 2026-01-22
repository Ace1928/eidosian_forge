from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlternativeOutputFormatValueValuesEnum(_messages.Enum):
    """Alternative output format to be generated based on the results of
    analysis.

    Values:
      ALTERNATIVE_OUTPUT_FORMAT_UNSPECIFIED: No alternative output format is
        specified.
      FHIR_BUNDLE: FHIR bundle output.
    """
    ALTERNATIVE_OUTPUT_FORMAT_UNSPECIFIED = 0
    FHIR_BUNDLE = 1