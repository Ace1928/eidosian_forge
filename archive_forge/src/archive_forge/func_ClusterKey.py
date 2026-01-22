from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from stdin.
def ClusterKey(cluster, key_type):
    """Return a cluster-generated public encryption key if there is one.

  Args:
    cluster: Cluster to check for an encryption key.
    key_type: Dataproc clusters publishes both RSA and ECIES public keys.

  Returns:
    The public key for the cluster if there is one, otherwise None
  """
    master_instance_refs = cluster.config.masterConfig.instanceReferences
    if not master_instance_refs:
        return None
    if key_type == 'ECIES':
        return master_instance_refs[0].publicEciesKey
    return master_instance_refs[0].publicKey