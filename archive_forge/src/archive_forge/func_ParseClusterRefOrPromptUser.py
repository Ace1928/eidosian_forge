from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def ParseClusterRefOrPromptUser(args):
    """Returns a ref to a GKE cluster based on args or prompting.

  Args:
    args: Parsed argument context object

  Returns:
    A Resource object representing the cluster

  Raises:
    ConfigurationError: when the user has not specified a cluster
      connection method and can't be prompted.
  """
    cluster_ref = args.CONCEPTS.cluster.Parse()
    if not cluster_ref:
        raise ConfigurationError('You must specify a cluster in a given location. Either use the `--cluster` and `--cluster-location` flags or set the kuberun/cluster and kuberun/cluster_location properties.')
    return cluster_ref