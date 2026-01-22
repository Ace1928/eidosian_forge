from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MultiChannelModeValueValuesEnum(_messages.Enum):
    """Mode for recognizing multi-channel audio.

    Values:
      MULTI_CHANNEL_MODE_UNSPECIFIED: Default value for the multi-channel
        mode. If the audio contains multiple channels, only the first channel
        will be transcribed; other channels will be ignored.
      SEPARATE_RECOGNITION_PER_CHANNEL: If selected, each channel in the
        provided audio is transcribed independently. This cannot be selected
        if the selected model is `latest_short`.
    """
    MULTI_CHANNEL_MODE_UNSPECIFIED = 0
    SEPARATE_RECOGNITION_PER_CHANNEL = 1