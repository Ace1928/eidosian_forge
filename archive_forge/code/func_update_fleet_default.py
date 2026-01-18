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
def update_fleet_default(self, default_cfg) -> None:
    """Update the feature configuration."""
    mask = ['fleet_default_member_config']
    feature = self.messages.Feature(name='notarealname')
    if default_cfg is not None:
        feature.fleetDefaultMemberConfig = self.messages.CommonFleetDefaultMemberConfigSpec(policycontroller=default_cfg)
    try:
        return self.Update(mask, feature)
    except gcloud_exceptions.Error as e:
        fne = self.FeatureNotEnabledError()
        if six.text_type(e) == six.text_type(fne):
            return self.Enable(feature)
        else:
            raise e