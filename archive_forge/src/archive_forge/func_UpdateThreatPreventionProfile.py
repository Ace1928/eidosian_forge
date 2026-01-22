from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def UpdateThreatPreventionProfile(self, existing_threat_prevention_profile_object, overrides, update_mask, operation_type):
    """Updates the existing threat_prevention_profile object.

    Args:
      existing_threat_prevention_profile_object: Existing Threat Prevention
        Profile JSON object.
      overrides: JSON object of overrides specified in command line.
      update_mask: String Arg specifying type of override which needs update.
      operation_type: String Arg specifying the type of operation which is
        performed in this method.

    Returns:
      Modified Threat Prevention Profile JSON object.
    """
    if operation_type == 'add_override':
        for override in overrides:
            does_override_exist, _ = self.CheckOverridesExist(existing_threat_prevention_profile_object, update_mask, override)
            if not does_override_exist:
                existing_threat_prevention_profile_object.get(update_mask).extend([override])
        return existing_threat_prevention_profile_object
    elif operation_type == 'update_override':
        for override in overrides:
            does_override_exist, override_index = self.CheckOverridesExist(existing_threat_prevention_profile_object, update_mask, override)
            if does_override_exist:
                existing_threat_prevention_profile_object.get(update_mask).pop(override_index)
                existing_threat_prevention_profile_object.get(update_mask).extend([override])
        return existing_threat_prevention_profile_object