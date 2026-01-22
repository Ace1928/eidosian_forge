from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportableContentsValueListEntryValuesEnum(_messages.Enum):
    """ExportableContentsValueListEntryValuesEnum enum type.

    Values:
      EXPORTABLE_CONTENT_UNSPECIFIED: Should not be used.
      ARTIFACT: Model artifact and any of its supported files. Will be
        exported to the location specified by the `artifactDestination` field
        of the ExportModelRequest.output_config object.
      IMAGE: The container image that is to be used when deploying this Model.
        Will be exported to the location specified by the `imageDestination`
        field of the ExportModelRequest.output_config object.
    """
    EXPORTABLE_CONTENT_UNSPECIFIED = 0
    ARTIFACT = 1
    IMAGE = 2