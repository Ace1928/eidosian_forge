from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.compute.sole_tenancy.node_templates import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def CreateNodeTemplate(node_template_ref, args, messages):
    """Creates a Node Template message from args."""
    node_affinity_labels = None
    if args.node_affinity_labels:
        node_affinity_labels = _ParseNodeAffinityLabels(args.node_affinity_labels, messages)
    node_type_flexbility = None
    if args.IsSpecified('node_requirements'):
        node_type_flexbility = messages.NodeTemplateNodeTypeFlexibility(cpus=six.text_type(args.node_requirements.get('vCPU', 'any')), localSsd=args.node_requirements.get('localSSD', None), memory=args.node_requirements.get('memory', 'any'))
    node_template = messages.NodeTemplate(name=node_template_ref.Name(), description=args.description, nodeAffinityLabels=node_affinity_labels, nodeType=args.node_type, nodeTypeFlexibility=node_type_flexbility)
    if args.IsSpecified('disk'):
        local_disk = messages.LocalDisk(diskCount=args.disk.get('count'), diskSizeGb=args.disk.get('size'), diskType=args.disk.get('type'))
        node_template.disks = [local_disk]
    if args.IsSpecified('cpu_overcommit_type'):
        overcommit_type = arg_utils.ChoiceToEnum(args.cpu_overcommit_type, messages.NodeTemplate.CpuOvercommitTypeValueValuesEnum)
        node_template.cpuOvercommitType = overcommit_type
    node_template.accelerators = GetAccelerators(args, messages)
    server_binding_flag = flags.GetServerBindingMapperFlag(messages)
    server_binding = messages.ServerBinding(type=server_binding_flag.GetEnumForChoice(args.server_binding))
    node_template.serverBinding = server_binding
    return node_template