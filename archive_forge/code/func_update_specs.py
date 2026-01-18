from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet import util as fleet_util
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.core import exceptions as gcloud_exceptions
import six
def update_specs(self, specs: SpecMapping) -> None:
    """Merges spec changes and sends and update to the API.

    Specs refer to PolicyControllerMembershipSpec objects defined here:
    third_party/py/googlecloudsdk/generated_clients/apis/gkehub/v1alpha/gkehub_v1alpha_messages.py

    (Note the above is for the ALPHA api track. Other tracks are found
    elsewhere.)

    Args:
      specs: Specs with updates. These are merged with the existing spec (new
        values overriding) and the merged result is sent to the Update api.

    Returns:
      None
    """
    feature = self.messages.Feature(membershipSpecs=self.hubclient.ToMembershipSpecs(specs))
    try:
        return self.Update(['membership_specs'], feature)
    except gcloud_exceptions.Error as e:
        fne = self.FeatureNotEnabledError()
        if six.text_type(e) == six.text_type(fne):
            return self.Enable(feature)
        else:
            raise e