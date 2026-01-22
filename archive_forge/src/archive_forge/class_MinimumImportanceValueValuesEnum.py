from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MinimumImportanceValueValuesEnum(_messages.Enum):
    """Filter to only get messages with importance >= level

    Values:
      JOB_MESSAGE_IMPORTANCE_UNKNOWN: The message importance isn't specified,
        or is unknown.
      JOB_MESSAGE_DEBUG: The message is at the 'debug' level: typically only
        useful for software engineers working on the code the job is running.
        Typically, Dataflow pipeline runners do not display log messages at
        this level by default.
      JOB_MESSAGE_DETAILED: The message is at the 'detailed' level: somewhat
        verbose, but potentially useful to users. Typically, Dataflow pipeline
        runners do not display log messages at this level by default. These
        messages are displayed by default in the Dataflow monitoring UI.
      JOB_MESSAGE_BASIC: The message is at the 'basic' level: useful for
        keeping track of the execution of a Dataflow pipeline. Typically,
        Dataflow pipeline runners display log messages at this level by
        default, and these messages are displayed by default in the Dataflow
        monitoring UI.
      JOB_MESSAGE_WARNING: The message is at the 'warning' level: indicating a
        condition pertaining to a job which may require human intervention.
        Typically, Dataflow pipeline runners display log messages at this
        level by default, and these messages are displayed by default in the
        Dataflow monitoring UI.
      JOB_MESSAGE_ERROR: The message is at the 'error' level: indicating a
        condition preventing a job from succeeding. Typically, Dataflow
        pipeline runners display log messages at this level by default, and
        these messages are displayed by default in the Dataflow monitoring UI.
    """
    JOB_MESSAGE_IMPORTANCE_UNKNOWN = 0
    JOB_MESSAGE_DEBUG = 1
    JOB_MESSAGE_DETAILED = 2
    JOB_MESSAGE_BASIC = 3
    JOB_MESSAGE_WARNING = 4
    JOB_MESSAGE_ERROR = 5