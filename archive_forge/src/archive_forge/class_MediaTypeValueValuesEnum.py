from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaTypeValueValuesEnum(_messages.Enum):
    """Specifies the kind of media held by assets of this asset type.

    Values:
      MEDIA_TYPE_UNSPECIFIED: AssetTypes with unspecified media types hold
        generic assets.
      VIDEO: AssetTypes with video media types have the following properties:
        1. Have a required and immutable metadata field called 'video_file' of
        type 'system:gcs-file', which is the path to a video file. 2. Support
        searching the content of the provided video asset.
    """
    MEDIA_TYPE_UNSPECIFIED = 0
    VIDEO = 1