from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class YarnApplication(_messages.Message):
    """A YARN application created by a job. Application information is a subset
  of org.apache.hadoop.yarn.proto.YarnProtos.ApplicationReportProto.Beta
  Feature: This report is available for testing purposes only. It may be
  changed before final release.

  Enums:
    StateValueValuesEnum: Required. The application state.

  Fields:
    name: Required. The application name.
    progress: Required. The numerical progress of the application, from 1 to
      100.
    state: Required. The application state.
    trackingUrl: Optional. The HTTP URL of the ApplicationMaster,
      HistoryServer, or TimelineServer that provides application-specific
      information. The URL uses the internal hostname, and requires a proxy
      server for resolution and, possibly, access.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Required. The application state.

    Values:
      STATE_UNSPECIFIED: Status is unspecified.
      NEW: Status is NEW.
      NEW_SAVING: Status is NEW_SAVING.
      SUBMITTED: Status is SUBMITTED.
      ACCEPTED: Status is ACCEPTED.
      RUNNING: Status is RUNNING.
      FINISHED: Status is FINISHED.
      FAILED: Status is FAILED.
      KILLED: Status is KILLED.
    """
        STATE_UNSPECIFIED = 0
        NEW = 1
        NEW_SAVING = 2
        SUBMITTED = 3
        ACCEPTED = 4
        RUNNING = 5
        FINISHED = 6
        FAILED = 7
        KILLED = 8
    name = _messages.StringField(1)
    progress = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    trackingUrl = _messages.StringField(4)