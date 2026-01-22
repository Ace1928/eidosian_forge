from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.networks.subnets import flags as subnet_flags
from googlecloudsdk.command_lib.compute.routers.nats import flags as nat_flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def UpdateNatMessage(nat, args, compute_holder):
    """Updates a NAT message with the specified arguments."""
    if args.subnet_option in [nat_flags.SubnetOption.ALL_RANGES, nat_flags.SubnetOption.PRIMARY_RANGES] or args.nat_custom_subnet_ip_ranges:
        ranges_to_nat, subnetworks = _ParseSubnetFields(args, compute_holder)
        nat.sourceSubnetworkIpRangesToNat = ranges_to_nat
        nat.subnetworks = subnetworks
    if args.nat_external_drain_ip_pool:
        drain_nat_ips = nat_flags.DRAIN_NAT_IP_ADDRESSES_ARG.ResolveAsResource(args, compute_holder.resources)
        nat.drainNatIps = [six.text_type(ip) for ip in drain_nat_ips]
        if not args.nat_external_ip_pool:
            nat.natIps = [ip for ip in nat.natIps if not _ContainIp(drain_nat_ips, ip)]
    if args.clear_nat_external_drain_ip_pool:
        nat.drainNatIps = []
    if args.auto_allocate_nat_external_ips or args.nat_external_ip_pool:
        option, nat_ips = _ParseNatIpFields(args, compute_holder)
        nat.natIpAllocateOption = option
        nat.natIps = nat_ips
    if args.auto_network_tier is not None:
        nat.autoNetworkTier = compute_holder.client.messages.RouterNat.AutoNetworkTierValueValuesEnum(args.auto_network_tier)
    if args.udp_idle_timeout is not None:
        nat.udpIdleTimeoutSec = args.udp_idle_timeout
    elif args.clear_udp_idle_timeout:
        nat.udpIdleTimeoutSec = None
    if args.icmp_idle_timeout is not None:
        nat.icmpIdleTimeoutSec = args.icmp_idle_timeout
    elif args.clear_icmp_idle_timeout:
        nat.icmpIdleTimeoutSec = None
    if args.tcp_established_idle_timeout is not None:
        nat.tcpEstablishedIdleTimeoutSec = args.tcp_established_idle_timeout
    elif args.clear_tcp_established_idle_timeout:
        nat.tcpEstablishedIdleTimeoutSec = None
    if args.tcp_transitory_idle_timeout is not None:
        nat.tcpTransitoryIdleTimeoutSec = args.tcp_transitory_idle_timeout
    elif args.clear_tcp_transitory_idle_timeout:
        nat.tcpTransitoryIdleTimeoutSec = None
    if args.tcp_time_wait_timeout is not None:
        nat.tcpTimeWaitTimeoutSec = args.tcp_time_wait_timeout
    elif args.clear_tcp_time_wait_timeout:
        nat.tcpTimeWaitTimeoutSec = None
    if args.min_ports_per_vm is not None:
        nat.minPortsPerVm = args.min_ports_per_vm
    elif args.clear_min_ports_per_vm:
        nat.minPortsPerVm = None
    if args.max_ports_per_vm is not None:
        nat.maxPortsPerVm = args.max_ports_per_vm
    elif args.clear_max_ports_per_vm:
        nat.maxPortsPerVm = None
    if args.enable_dynamic_port_allocation is not None:
        nat.enableDynamicPortAllocation = args.enable_dynamic_port_allocation
    if args.enable_logging is not None or args.log_filter is not None:
        nat.logConfig = nat.logConfig or compute_holder.client.messages.RouterNatLogConfig()
    if args.enable_logging is not None:
        nat.logConfig.enable = args.enable_logging
    if args.log_filter is not None:
        nat.logConfig.filter = _TranslateLogFilter(args.log_filter, compute_holder)
    if args.enable_endpoint_independent_mapping is not None:
        nat.enableEndpointIndependentMapping = args.enable_endpoint_independent_mapping
    if args.rules:
        nat.rules = _ParseRulesFromYamlFile(args.rules, compute_holder)
    return nat