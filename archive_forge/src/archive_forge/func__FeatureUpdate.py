from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from typing import Iterator
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet import resources as fleet_resources
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as fleet_messages
def _FeatureUpdate(self) -> fleet_messages.FeatureUpdate:
    """Constructs message FeatureUpdate."""
    feature_update = fleet_messages.FeatureUpdate()
    feature_update.securityPostureConfig = self._SecurityPostureConfig()
    feature_update.binaryAuthorizationConfig = self._BinaryAuthorzationConfig()
    return self.TrimEmpty(feature_update)