from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InternalChecker(_messages.Message):
    """An internal checker allows Uptime checks to run on private/internal GCP
  resources.

  Enums:
    StateValueValuesEnum: The current operational state of the internal
      checker.

  Fields:
    displayName: The checker's human-readable name. The display name should be
      unique within a Cloud Monitoring Metrics Scope in order to make it
      easier to identify; however, uniqueness is not enforced.
    gcpZone: The GCP zone the Uptime check should egress from. Only respected
      for internal Uptime checks, where internal_network is specified.
    name: A unique resource name for this InternalChecker. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/internalCheckers/[INTERNAL_CHECKER_ID]
      [PROJECT_ID_OR_NUMBER] is the Cloud Monitoring Metrics Scope project for
      the Uptime check config associated with the internal checker.
    network: The GCP VPC network (https://cloud.google.com/vpc/docs/vpc) where
      the internal resource lives (ex: "default").
    peerProjectId: The GCP project ID where the internal checker lives. Not
      necessary the same as the Metrics Scope project.
    state: The current operational state of the internal checker.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The current operational state of the internal checker.

    Values:
      UNSPECIFIED: An internal checker should never be in the unspecified
        state.
      CREATING: The checker is being created, provisioned, and configured. A
        checker in this state can be returned by ListInternalCheckers or
        GetInternalChecker, as well as by examining the long running Operation
        (https://cloud.google.com/apis/design/design_patterns#long_running_ope
        rations) that created it.
      RUNNING: The checker is running and available for use. A checker in this
        state can be returned by ListInternalCheckers or GetInternalChecker as
        well as by examining the long running Operation (https://cloud.google.
        com/apis/design/design_patterns#long_running_operations) that created
        it. If a checker is being torn down, it is neither visible nor usable,
        so there is no "deleting" or "down" state.
    """
        UNSPECIFIED = 0
        CREATING = 1
        RUNNING = 2
    displayName = _messages.StringField(1)
    gcpZone = _messages.StringField(2)
    name = _messages.StringField(3)
    network = _messages.StringField(4)
    peerProjectId = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)