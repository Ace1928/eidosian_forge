from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContentStructureValueValuesEnum(_messages.Enum):
    """The content structure in the source location. If not specified, the
    server treats the input source files as BUNDLE.

    Values:
      CONTENT_STRUCTURE_UNSPECIFIED: If the content structure is not
        specified, the default value `BUNDLE` is used.
      BUNDLE: The source file contains one or more lines of newline-delimited
        JSON (ndjson). Each line is a bundle that contains one or more
        resources.
      RESOURCE: The source file contains one or more lines of newline-
        delimited JSON (ndjson). Each line is a single resource.
      BUNDLE_PRETTY: The entire file is one JSON bundle. The JSON can span
        multiple lines.
      RESOURCE_PRETTY: The entire file is one JSON resource. The JSON can span
        multiple lines.
    """
    CONTENT_STRUCTURE_UNSPECIFIED = 0
    BUNDLE = 1
    RESOURCE = 2
    BUNDLE_PRETTY = 3
    RESOURCE_PRETTY = 4