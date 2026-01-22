from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as sdk_core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def CreateNodeSpec(api_version):
    """Creates the repeated structure nodeSpec from args."""

    def Process(ref, args, request):
        tpu_messages = GetMessagesModule(api_version)
        if request.queuedResource is None:
            request.queuedResource = tpu_messages.QueuedResource()
        if request.queuedResource.tpu is None:
            request.queuedResource.tpu = tpu_messages.Tpu()
        if request.queuedResource.tpu.nodeSpec:
            node_spec = request.queuedResource.tpu.nodeSpec[0]
        else:
            request.queuedResource.tpu.nodeSpec = []
            node_spec = tpu_messages.NodeSpec()
            node_spec.node = tpu_messages.Node()
        node_spec.parent = ref.Parent().RelativeName()
        if args.accelerator_type:
            node_spec.node.acceleratorType = args.accelerator_type
        node_spec.node.runtimeVersion = args.runtime_version
        if args.data_disk:
            node_spec.node.dataDisks = []
            for data_disk in args.data_disk:
                attached_disk = tpu_messages.AttachedDisk(sourceDisk=data_disk.sourceDisk, mode=data_disk.mode)
                node_spec.node.dataDisks.append(attached_disk)
        if args.description:
            node_spec.node.description = args.description
        if args.labels:
            node_spec.node.labels = tpu_messages.Node.LabelsValue()
            node_spec.node.labels.additionalProperties = []
            for key, value in args.labels.items():
                node_spec.node.labels.additionalProperties.append(tpu_messages.Node.LabelsValue.AdditionalProperty(key=key, value=value))
        if args.range:
            node_spec.node.cidrBlock = args.range
        if args.shielded_secure_boot:
            node_spec.node.shieldedInstanceConfig = tpu_messages.ShieldedInstanceConfig(enableSecureBoot=True)
        if api_version == 'v2alpha1' and args.autocheckpoint_enabled:
            node_spec.node.autocheckpointEnabled = True
        node_spec.node.networkConfig = tpu_messages.NetworkConfig()
        node_spec.node.serviceAccount = tpu_messages.ServiceAccount()
        if args.network:
            node_spec.node.networkConfig.network = args.network
        if args.subnetwork:
            node_spec.node.networkConfig.subnetwork = args.subnetwork
        if args.service_account:
            node_spec.node.serviceAccount.email = args.service_account
        if args.scopes:
            node_spec.node.serviceAccount.scope = args.scopes
        if args.tags:
            node_spec.node.tags = args.tags
        node_spec.node.networkConfig.enableExternalIps = not args.internal_ips
        if api_version == 'v2alpha1' and args.boot_disk:
            node_spec.node.bootDiskConfig = ParseBootDiskConfig(args.boot_disk)
        node_spec.node.metadata = MergeMetadata(args, api_version)
        if args.node_prefix and (not args.node_count):
            raise exceptions.ConflictingArgumentsException('Can only specify --node-prefix if --node-count is also specified.')
        if args.node_id:
            node_spec.nodeId = args.node_id
        elif args.node_count:
            if api_version == 'v2alpha1':
                node_spec.multiNodeParams = tpu_messages.MultiNodeParams()
                node_spec.multiNodeParams.nodeCount = args.node_count
                if args.node_prefix:
                    node_spec.multiNodeParams.nodeIdPrefix = args.node_prefix
            else:
                node_spec.multisliceParams = tpu_messages.MultisliceParams()
                node_spec.multisliceParams.nodeCount = args.node_count
                if args.node_prefix:
                    node_spec.multisliceParams.nodeIdPrefix = args.node_prefix
        request.queuedResource.tpu.nodeSpec = [node_spec]
        return request
    return Process