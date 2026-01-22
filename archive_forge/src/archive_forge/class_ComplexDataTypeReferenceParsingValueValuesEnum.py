from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComplexDataTypeReferenceParsingValueValuesEnum(_messages.Enum):
    """Enable parsing of references within complex FHIR data types such as
    Extensions. If this value is set to ENABLED, then features like
    referential integrity and Bundle reference rewriting apply to all
    references. If this flag has not been specified the behavior of the FHIR
    store will not change, references in complex data types will not be
    parsed. New stores will have this value set to ENABLED after a
    notification period. Warning: turning on this flag causes processing
    existing resources to fail if they contain references to non-existent
    resources.

    Values:
      COMPLEX_DATA_TYPE_REFERENCE_PARSING_UNSPECIFIED: No parsing behavior
        specified. This is the same as DISABLED for backwards compatibility.
      DISABLED: References in complex data types are ignored.
      ENABLED: References in complex data types are parsed.
    """
    COMPLEX_DATA_TYPE_REFERENCE_PARSING_UNSPECIFIED = 0
    DISABLED = 1
    ENABLED = 2