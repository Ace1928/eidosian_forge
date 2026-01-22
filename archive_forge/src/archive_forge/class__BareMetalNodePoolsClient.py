from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import bare_metal_clusters as clusters
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
class _BareMetalNodePoolsClient(clusters.ClustersClient):
    """Base class for GKE OnPrem Bare Metal API clients."""

    def _node_taints(self, args: parser_extensions.Namespace):
        """Constructs proto message NodeTaint."""
        taint_messages = []
        node_taints = getattr(args, 'node_taints', {})
        if not node_taints:
            return []
        for node_taint in node_taints.items():
            taint_object = self._parse_node_taint(node_taint)
            taint_messages.append(messages.NodeTaint(**taint_object))
        return taint_messages

    def _node_labels(self, args: parser_extensions.Namespace):
        """Constructs proto message LabelsValue."""
        node_labels = getattr(args, 'node_labels', {})
        additional_property_messages = []
        if not node_labels:
            return None
        for key, value in node_labels.items():
            additional_property_messages.append(messages.BareMetalNodePoolConfig.LabelsValue.AdditionalProperty(key=key, value=value))
        labels_value_message = messages.BareMetalNodePoolConfig.LabelsValue(additionalProperties=additional_property_messages)
        return labels_value_message

    def _node_configs_from_file(self, args: parser_extensions.Namespace):
        """Constructs proto message field node_configs."""
        if not args.node_configs_from_file:
            return []
        node_configs = args.node_configs_from_file.get('nodeConfigs', [])
        if not node_configs:
            raise exceptions.BadArgumentException('--node_configs_from_file', 'Missing field [nodeConfigs] in Node configs file.')
        node_config_messages = []
        for node_config in node_configs:
            node_config_messages.append(self._bare_metal_node_config(node_config))
        return node_config_messages

    def _bare_metal_node_config(self, node_config):
        """Constructs proto message BareMetalNodeConfig."""
        node_ip = node_config.get('nodeIP', '')
        if not node_ip:
            raise exceptions.BadArgumentException('--node_configs_from_file', 'Missing field [nodeIP] in Node configs file.')
        kwargs = {'nodeIp': node_ip, 'labels': self._node_config_labels(node_config.get('labels', {}))}
        return messages.BareMetalNodeConfig(**kwargs)

    def _node_config_labels(self, labels):
        """Constructs proto message LabelsValue."""
        additional_property_messages = []
        if not labels:
            return None
        for key, value in labels.items():
            additional_property_messages.append(messages.BareMetalNodeConfig.LabelsValue.AdditionalProperty(key=key, value=value))
        labels_value_message = messages.BareMetalNodeConfig.LabelsValue(additionalProperties=additional_property_messages)
        return labels_value_message

    def _node_configs_from_flag(self, args: parser_extensions.Namespace):
        """Constructs proto message field node_configs."""
        node_configs = []
        node_config_flag_value = getattr(args, 'node_configs', None)
        if node_config_flag_value:
            for node_config in node_config_flag_value:
                node_configs.append(self.node_config(node_config))
        return node_configs

    def _serialized_image_pulls_disabled(self, args: parser_extensions.Namespace):
        if 'disable_serialize_image_pulls' in args.GetSpecifiedArgsDict():
            return True
        elif 'enable_serialize_image_pulls' in args.GetSpecifiedArgsDict():
            return False
        else:
            return None

    def _kubelet_config(self, args: parser_extensions.Namespace):
        kwargs = {'registryPullQps': self.GetFlag(args, 'registry_pull_qps'), 'registryBurst': self.GetFlag(args, 'registry_burst'), 'serializeImagePullsDisabled': self._serialized_image_pulls_disabled(args)}
        if any(kwargs.values()):
            return messages.BareMetalKubeletConfig(**kwargs)
        return None

    def _node_pool_config(self, args: parser_extensions.Namespace):
        """Constructs proto message BareMetalNodePoolConfig."""
        if 'node_configs_from_file' in args.GetSpecifiedArgsDict():
            node_configs = self._node_configs_from_file(args)
        else:
            node_configs = self._node_configs_from_flag(args)
        kwargs = {'nodeConfigs': node_configs, 'labels': self._node_labels(args), 'taints': self._node_taints(args), 'kubeletConfig': self._kubelet_config(args)}
        if any(kwargs.values()):
            return messages.BareMetalNodePoolConfig(**kwargs)
        return None

    def _annotations(self, args: parser_extensions.Namespace):
        """Constructs proto message AnnotationsValue."""
        annotations = getattr(args, 'annotations', {})
        additional_property_messages = []
        if not annotations:
            return None
        for key, value in annotations.items():
            additional_property_messages.append(messages.BareMetalNodePool.AnnotationsValue.AdditionalProperty(key=key, value=value))
        annotation_value_message = messages.BareMetalNodePool.AnnotationsValue(additionalProperties=additional_property_messages)
        return annotation_value_message

    def _bare_metal_node_pool(self, args: parser_extensions.Namespace):
        """Constructs proto message BareMetalNodePool."""
        kwargs = {'name': self._node_pool_name(args), 'nodePoolConfig': self._node_pool_config(args), 'displayName': getattr(args, 'display_name', None), 'annotations': self._annotations(args), 'bareMetalVersion': getattr(args, 'version', None)}
        return messages.BareMetalNodePool(**kwargs)