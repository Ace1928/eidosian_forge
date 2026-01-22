from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core.util import text
def AddClusterConfig(self):
    """Add the cluster-config argument as repeated kv dicts."""
    self.parser.add_argument('--cluster-config', action='append', type=arg_parsers.ArgDict(spec={'id': str, 'zone': str, 'nodes': int, 'kms-key': str, 'autoscaling-min-nodes': int, 'autoscaling-max-nodes': int, 'autoscaling-cpu-target': int, 'autoscaling-storage-target': int}, required_keys=['id', 'zone'], max_length=8), metavar='id=ID,zone=ZONE,nodes=NODES,kms-key=KMS_KEY,autoscaling-min-nodes=AUTOSCALING_MIN_NODES,autoscaling-max-nodes=AUTOSCALING_MAX_NODES,autoscaling-cpu-target=AUTOSCALING_CPU_TARGET,autoscaling-storage-target=AUTOSCALING_STORAGE_TARGET', help=textwrap.dedent('        *Repeatable*. Specify cluster config as a key-value dictionary.\n\n        This is the recommended argument for specifying cluster configurations.\n\n        Keys can be:\n\n          *id*: Required. The ID of the cluster.\n\n          *zone*: Required. ID of the zone where the cluster is located. Supported zones are listed at https://cloud.google.com/bigtable/docs/locations.\n\n          *nodes*: The number of nodes in the cluster. Default=1.\n\n          *kms-key*: The Cloud KMS (Key Management Service) cryptokey that will be used to protect the cluster.\n\n          *autoscaling-min-nodes*: The minimum number of nodes for autoscaling.\n\n          *autoscaling-max-nodes*: The maximum number of nodes for autoscaling.\n\n          *autoscaling-cpu-target*: The target CPU utilization percentage for autoscaling. Accepted values are from 10 to 80.\n\n          *autoscaling-storage-target*: The target storage utilization gibibytes per node for autoscaling. Accepted values are from 2560 to 5120 for SSD clusters and 8192 to 16384 for HDD clusters.\n\n        If this argument is specified, the deprecated arguments for configuring a single cluster will be ignored, including *--cluster*, *--cluster-zone*, *--cluster-num-nodes*.\n\n        See *EXAMPLES* section.\n        '))
    return self