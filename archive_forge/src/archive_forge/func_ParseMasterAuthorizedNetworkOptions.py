from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def ParseMasterAuthorizedNetworkOptions(self, options, cluster):
    """Parses the options for master authorized networks."""
    if options.master_authorized_networks and (not options.enable_master_authorized_networks):
        raise util.Error(MISMATCH_AUTHORIZED_NETWORKS_ERROR_MSG)
    elif options.enable_master_authorized_networks is None:
        cluster.masterAuthorizedNetworksConfig = None
    elif not options.enable_master_authorized_networks:
        authorized_networks = self.messages.MasterAuthorizedNetworksConfig(enabled=False)
        cluster.masterAuthorizedNetworksConfig = authorized_networks
    else:
        authorized_networks = self.messages.MasterAuthorizedNetworksConfig(enabled=options.enable_master_authorized_networks)
        if options.master_authorized_networks:
            for network in options.master_authorized_networks:
                authorized_networks.cidrBlocks.append(self.messages.CidrBlock(cidrBlock=network))
        cluster.masterAuthorizedNetworksConfig = authorized_networks
    if options.enable_google_cloud_access is not None:
        if cluster.masterAuthorizedNetworksConfig is None:
            cluster.masterAuthorizedNetworksConfig = self.messages.MasterAuthorizedNetworksConfig(enabled=False)
        cluster.masterAuthorizedNetworksConfig.gcpPublicCidrsAccessEnabled = options.enable_google_cloud_access