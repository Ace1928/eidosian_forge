from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PostStartupScriptBehaviorValueValuesEnum(_messages.Enum):
    """Behavior for the post startup script.

    Values:
      POST_STARTUP_SCRIPT_BEHAVIOR_UNSPECIFIED: Unspecified post startup
        script behavior. Will run only once at creation.
      RUN_EVERY_START: Runs the post startup script provided during creation
        at every start.
      DOWNLOAD_AND_RUN_EVERY_START: Downloads and runs the provided post
        startup script at every start.
    """
    POST_STARTUP_SCRIPT_BEHAVIOR_UNSPECIFIED = 0
    RUN_EVERY_START = 1
    DOWNLOAD_AND_RUN_EVERY_START = 2