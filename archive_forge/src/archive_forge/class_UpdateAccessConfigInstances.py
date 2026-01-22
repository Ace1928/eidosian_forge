from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.GA)
class UpdateAccessConfigInstances(base.UpdateCommand):
    """Update a Compute Engine virtual machine access configuration."""
    _support_public_dns = False
    _support_network_tier = False

    @classmethod
    def Args(cls, parser):
        _Args(parser, support_public_dns=cls._support_public_dns, support_network_tier=cls._support_network_tier)

    def CreateReference(self, client, resources, args):
        return flags.INSTANCE_ARG.ResolveAsResource(args, resources, scope_lister=flags.GetInstanceZoneScopeLister(client))

    def CreateV4AddressConfig(self, client):
        return client.messages.AccessConfig(type=client.messages.AccessConfig.TypeValueValuesEnum.ONE_TO_ONE_NAT)

    def CreateV6AddressConfig(self, client):
        return client.messages.AccessConfig(type=client.messages.AccessConfig.TypeValueValuesEnum.DIRECT_IPV6)

    def GetGetRequest(self, client, instance_ref):
        return (client.apitools_client.instances, 'Get', client.messages.ComputeInstancesGetRequest(**instance_ref.AsDict()))

    def GetUpdateRequest(self, client, args, instance_ref, access_config):
        return (client.apitools_client.instances, 'UpdateAccessConfig', client.messages.ComputeInstancesUpdateAccessConfigRequest(instance=instance_ref.instance, networkInterface=args.network_interface, accessConfig=access_config, project=instance_ref.project, zone=instance_ref.zone))

    def Modify(self, client, args, original):
        set_public_dns = None
        if self._support_public_dns:
            if args.public_dns:
                set_public_dns = True
            elif args.no_public_dns:
                set_public_dns = False
        set_ptr = None
        if args.public_ptr:
            set_ptr = True
        elif args.no_public_ptr:
            set_ptr = False
        set_ipv6_ptr = None
        if args.ipv6_public_ptr_domain:
            set_ipv6_ptr = True
        elif args.no_ipv6_public_ptr:
            set_ipv6_ptr = False
        if set_ptr is not None and set_ipv6_ptr is not None:
            raise exceptions.InvalidArgumentException('--ipv6-public-ptr-domain', 'Cannot update --public-ptr-domain and --ipv6-public-ptr-domain at same time.')
        if self._support_public_dns and set_public_dns is not None:
            access_config = self.CreateV4AddressConfig(client)
            access_config.setPublicDns = set_public_dns
            return access_config
        if set_ptr is not None:
            access_config = self.CreateV4AddressConfig(client)
            access_config.setPublicPtr = set_ptr
            new_ptr = '' if args.public_ptr_domain is None else args.public_ptr_domain
            access_config.publicPtrDomainName = new_ptr
            return access_config
        if set_ipv6_ptr is not None:
            access_config = self.CreateV6AddressConfig(client)
            new_ptr = '' if args.ipv6_public_ptr_domain is None else args.ipv6_public_ptr_domain
            access_config.publicPtrDomainName = new_ptr
            return access_config
        if self._support_network_tier and args.network_tier is not None:
            access_config = self.CreateV4AddressConfig(client)
            access_config.networkTier = client.messages.AccessConfig.NetworkTierValueValuesEnum(args.network_tier)
            return access_config
        return None

    def Run(self, args):
        flags.ValidatePublicPtrFlags(args)
        flags.ValidateIpv6PublicPtrFlags(args)
        if self._support_public_dns:
            flags.ValidatePublicDnsFlags(args)
        if self._support_network_tier:
            flags.ValidateNetworkTierArgs(args)
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        instance_ref = self.CreateReference(client, holder.resources, args)
        get_request = self.GetGetRequest(client, instance_ref)
        objects = client.MakeRequests([get_request])
        new_access_config = self.Modify(client, args, objects[0])
        if new_access_config is None:
            log.status.Print('No change requested; skipping update for [{0}].'.format(objects[0].name))
            return objects
        return client.MakeRequests(requests=[self.GetUpdateRequest(client, args, instance_ref, new_access_config)])