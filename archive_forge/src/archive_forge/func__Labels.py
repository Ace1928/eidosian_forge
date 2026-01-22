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
def _Labels(self) -> fleet_messages.Rollout.LabelsValue:
    """Parses --labels."""
    if '--labels' not in self.args.GetSpecifiedArgs():
        return None
    labels = self.args.labels
    labels_value = fleet_messages.Rollout.LabelsValue()
    for key, value in labels.items():
        labels_value.additionalProperties.append(fleet_messages.Rollout.LabelsValue.AdditionalProperty(key=key, value=value))
    return labels_value