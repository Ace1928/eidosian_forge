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
def ParseIPAliasOptions(self, options, cluster):
    """Parses the options for IP Alias."""
    ip_alias_only_options = [('services-ipv4-cidr', options.services_ipv4_cidr), ('create-subnetwork', options.create_subnetwork), ('cluster-secondary-range-name', options.cluster_secondary_range_name), ('services-secondary-range-name', options.services_secondary_range_name), ('disable-pod-cidr-overprovision', options.disable_pod_cidr_overprovision), ('stack-type', options.stack_type), ('ipv6-access-type', options.ipv6_access_type)]
    if not options.enable_ip_alias:
        for name, opt in ip_alias_only_options:
            if opt:
                raise util.Error(PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-ip-alias', opt=name))
    if options.subnetwork and options.create_subnetwork is not None:
        raise util.Error(CREATE_SUBNETWORK_WITH_SUBNETWORK_ERROR_MSG)
    if options.enable_ip_alias:
        subnetwork_name = None
        node_ipv4_cidr = None
        if options.create_subnetwork is not None:
            for key in options.create_subnetwork:
                if key not in ['name', 'range']:
                    raise util.Error(CREATE_SUBNETWORK_INVALID_KEY_ERROR_MSG.format(key=key))
            subnetwork_name = options.create_subnetwork.get('name', None)
            node_ipv4_cidr = options.create_subnetwork.get('range', None)
        policy = self.messages.IPAllocationPolicy(useIpAliases=options.enable_ip_alias, createSubnetwork=options.create_subnetwork is not None, subnetworkName=subnetwork_name, clusterIpv4CidrBlock=options.cluster_ipv4_cidr, nodeIpv4CidrBlock=node_ipv4_cidr, servicesIpv4CidrBlock=options.services_ipv4_cidr, clusterSecondaryRangeName=options.cluster_secondary_range_name, servicesSecondaryRangeName=options.services_secondary_range_name)
        if options.disable_pod_cidr_overprovision is not None:
            policy.podCidrOverprovisionConfig = self.messages.PodCIDROverprovisionConfig(disable=options.disable_pod_cidr_overprovision)
        if options.tpu_ipv4_cidr:
            policy.tpuIpv4CidrBlock = options.tpu_ipv4_cidr
        if options.stack_type is not None:
            policy.stackType = util.GetCreateStackTypeMapper(self.messages).GetEnumForChoice(options.stack_type)
        if options.ipv6_access_type is not None:
            policy.ipv6AccessType = util.GetIpv6AccessTypeMapper(self.messages).GetEnumForChoice(options.ipv6_access_type)
        cluster.clusterIpv4Cidr = None
        cluster.ipAllocationPolicy = policy
    elif options.enable_ip_alias is not None:
        cluster.ipAllocationPolicy = self.messages.IPAllocationPolicy(useRoutes=True)
    return cluster