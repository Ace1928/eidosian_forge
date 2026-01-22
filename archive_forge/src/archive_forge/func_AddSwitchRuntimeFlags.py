from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.compute.networks import flags as compute_network_flags
from googlecloudsdk.command_lib.compute.networks.subnets import flags as compute_subnet_flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.notebooks import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddSwitchRuntimeFlags(api_version, parser):
    """Adds accelerator and machine type flags to the parser for switch."""
    accelerator_choices = ['NVIDIA_TESLA_A100', 'NVIDIA_TESLA_K80', 'NVIDIA_TESLA_P100', 'NVIDIA_TESLA_V100', 'NVIDIA_TESLA_P4', 'NVIDIA_TESLA_T4', 'NVIDIA_TESLA_T4_VWS', 'NVIDIA_TESLA_P100_VWS', 'NVIDIA_TESLA_P4_VWS', 'TPU_V2', 'TPU_V3']
    AddRuntimeResource(api_version, parser)
    parser.add_argument('--machine-type', help='machine type')
    accelerator_config_group = parser.add_group()
    accelerator_config_group.add_argument('--accelerator-type', help='Type of this accelerator.', choices=accelerator_choices, default=None)
    accelerator_config_group.add_argument('--accelerator-core-count', help='Count of cores of this accelerator.', type=int)