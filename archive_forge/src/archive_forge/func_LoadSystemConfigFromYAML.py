from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
def LoadSystemConfigFromYAML(node_config, content, opt_readonly_port_flag, messages):
    """Load system configuration (sysctl & kubelet config) from YAML/JSON file.

  Args:
    node_config: The node config object to be populated.
    content: The YAML/JSON string that contains sysctl and kubelet options.
    opt_readonly_port_flag: kubelet readonly port enabled.
    messages: The message module.

  Raises:
    Error: when there's any errors on parsing the YAML/JSON system config.
  """
    try:
        opts = yaml.load(content)
    except yaml.YAMLParseError as e:
        raise NodeConfigError('config is not valid YAML/JSON: {0}'.format(e))
    _CheckNodeConfigFields('<root>', opts, {NC_KUBELET_CONFIG: dict, NC_LINUX_CONFIG: dict})
    kubelet_config_opts = opts.get(NC_KUBELET_CONFIG)
    if kubelet_config_opts:
        config_fields = {NC_CPU_MANAGER_POLICY: str, NC_CPU_CFS_QUOTA: bool, NC_CPU_CFS_QUOTA_PERIOD: str, NC_POD_PIDS_LIMIT: int, NC_KUBELET_READONLY_PORT: bool}
        _CheckNodeConfigFields(NC_KUBELET_CONFIG, kubelet_config_opts, config_fields)
        node_config.kubeletConfig = messages.NodeKubeletConfig()
        node_config.kubeletConfig.cpuManagerPolicy = kubelet_config_opts.get(NC_CPU_MANAGER_POLICY)
        node_config.kubeletConfig.cpuCfsQuota = kubelet_config_opts.get(NC_CPU_CFS_QUOTA)
        node_config.kubeletConfig.cpuCfsQuotaPeriod = kubelet_config_opts.get(NC_CPU_CFS_QUOTA_PERIOD)
        node_config.kubeletConfig.podPidsLimit = kubelet_config_opts.get(NC_POD_PIDS_LIMIT)
        node_config.kubeletConfig.insecureKubeletReadonlyPortEnabled = kubelet_config_opts.get(NC_KUBELET_READONLY_PORT)
    ro_in_cfg = node_config is not None and node_config.kubeletConfig is not None and (node_config.kubeletConfig.insecureKubeletReadonlyPortEnabled is not None)
    ro_in_flag = opt_readonly_port_flag is not None
    if ro_in_cfg and ro_in_flag:
        raise NodeConfigError(INVALID_NC_FLAG_CONFIG_OVERLAP)
    linux_config_opts = opts.get(NC_LINUX_CONFIG)
    if linux_config_opts:
        _CheckNodeConfigFields(NC_LINUX_CONFIG, linux_config_opts, {NC_SYSCTL: dict, NC_CGROUP_MODE: str, NC_HUGEPAGE: dict})
        node_config.linuxNodeConfig = messages.LinuxNodeConfig()
        sysctl_opts = linux_config_opts.get(NC_SYSCTL)
        if sysctl_opts:
            node_config.linuxNodeConfig.sysctls = node_config.linuxNodeConfig.SysctlsValue()
            for key, value in sorted(six.iteritems(sysctl_opts)):
                _CheckNodeConfigValueType(key, value, str)
                node_config.linuxNodeConfig.sysctls.additionalProperties.append(node_config.linuxNodeConfig.sysctls.AdditionalProperty(key=key, value=value))
        cgroup_mode_opts = linux_config_opts.get(NC_CGROUP_MODE)
        if cgroup_mode_opts:
            if not hasattr(messages.LinuxNodeConfig, 'cgroupMode'):
                raise NodeConfigError('setting cgroupMode as {0} is not supported'.format(cgroup_mode_opts))
            cgroup_mode_mapping = {'CGROUP_MODE_UNSPECIFIED': messages.LinuxNodeConfig.CgroupModeValueValuesEnum.CGROUP_MODE_UNSPECIFIED, 'CGROUP_MODE_V1': messages.LinuxNodeConfig.CgroupModeValueValuesEnum.CGROUP_MODE_V1, 'CGROUP_MODE_V2': messages.LinuxNodeConfig.CgroupModeValueValuesEnum.CGROUP_MODE_V2}
            if cgroup_mode_opts not in cgroup_mode_mapping:
                raise NodeConfigError('cgroup mode "{0}" is not supported, the supported options are CGROUP_MODE_UNSPECIFIED, CGROUP_MODE_V1, CGROUP_MODE_V2'.format(cgroup_mode_opts))
            node_config.linuxNodeConfig.cgroupMode = cgroup_mode_mapping[cgroup_mode_opts]
        hugepage_opts = linux_config_opts.get(NC_HUGEPAGE)
        if hugepage_opts:
            node_config.linuxNodeConfig.hugepages = messages.HugepagesConfig()
            hugepage_size2m = hugepage_opts.get(NC_HUGEPAGE_2M)
            if hugepage_size2m:
                node_config.linuxNodeConfig.hugepages.hugepageSize2m = hugepage_size2m
            hugepage_size1g = hugepage_opts.get(NC_HUGEPAGE_1G)
            if hugepage_size1g:
                node_config.linuxNodeConfig.hugepages.hugepageSize1g = hugepage_size1g