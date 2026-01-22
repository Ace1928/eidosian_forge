from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Criteria(_messages.Message):
    """Criteria specific to the AlertPolicys that this Snooze applies to. The
  Snooze will suppress alerts that come from one of the AlertPolicys whose
  names are supplied.

  Fields:
    policies: The specific AlertPolicy names for the alert that should be
      snoozed. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/alertPolicies/[POLICY_ID] There is a
      limit of 16 policies per snooze. This limit is checked during snooze
      creation.
  """
    policies = _messages.StringField(1, repeated=True)