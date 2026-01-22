from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def GetSecurityProfileEntities(self, name):
    """Calls the Security Profile Get API to return the threat prevention profile object.

    Args:
      name: Fully specified Security Profile.

    Returns:
      An etag and a Dict of existing Threat Prevention Profile configuration.
    """
    response = self.GetSecurityProfile(name)
    if response.threatPreventionProfile is None:
        return (response.etag, {'severityOverrides': [], 'threatOverrides': []})
    else:
        profile = encoding.MessageToDict(response.threatPreventionProfile)
        if not any(profile):
            return (response.etag, {'severityOverrides': [], 'threatOverrides': []})
        else:
            if profile.get('severityOverrides') is None:
                profile['severityOverrides'] = []
            if profile.get('threatOverrides') is None:
                profile['threatOverrides'] = []
            return (response.etag, profile)