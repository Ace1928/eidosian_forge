from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileSetSpecTypeValueValuesEnum(_messages.Enum):
    """Optional. Specifies how source URIs are interpreted for constructing
    the file set to load. By default, source URIs are expanded against the
    underlying storage. You can also specify manifest files to control how the
    file set is constructed. This option is only applicable to object storage
    systems.

    Values:
      FILE_SET_SPEC_TYPE_FILE_SYSTEM_MATCH: This option expands source URIs by
        listing files from the object store. It is the default behavior if
        FileSetSpecType is not set.
      FILE_SET_SPEC_TYPE_NEW_LINE_DELIMITED_MANIFEST: This option indicates
        that the provided URIs are newline-delimited manifest files, with one
        URI per line. Wildcard URIs are not supported.
    """
    FILE_SET_SPEC_TYPE_FILE_SYSTEM_MATCH = 0
    FILE_SET_SPEC_TYPE_NEW_LINE_DELIMITED_MANIFEST = 1