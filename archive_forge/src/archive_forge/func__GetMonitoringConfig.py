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
def _GetMonitoringConfig(options, messages):
    """Gets the MonitoringConfig from create and update options."""
    comp = None
    prom = None
    adv_obs = None
    config = messages.MonitoringConfig()
    if options.enable_managed_prometheus is not None:
        prom = messages.ManagedPrometheusConfig(enabled=options.enable_managed_prometheus)
        config.managedPrometheusConfig = prom
    if hasattr(options, 'disable_managed_prometheus'):
        if options.disable_managed_prometheus is not None:
            prom = messages.ManagedPrometheusConfig(enabled=not options.disable_managed_prometheus)
            config.managedPrometheusConfig = prom
    if options.monitoring is not None:
        if any((c not in MONITORING_OPTIONS for c in options.monitoring)):
            raise util.Error('[' + ', '.join(options.monitoring) + '] contains option(s) that are not supported for monitoring.')
        comp = messages.MonitoringComponentConfig()
        if NONE in options.monitoring:
            if len(options.monitoring) > 1:
                raise util.Error('Cannot include other values when None is specified.')
            else:
                config.componentConfig = comp
                return config
        if SYSTEM not in options.monitoring:
            raise util.Error('Must include system monitoring if any monitoring is enabled.')
        comp.enableComponents.append(messages.MonitoringComponentConfig.EnableComponentsValueListEntryValuesEnum.SYSTEM_COMPONENTS)
        if WORKLOAD in options.monitoring:
            comp.enableComponents.append(messages.MonitoringComponentConfig.EnableComponentsValueListEntryValuesEnum.WORKLOADS)
        if API_SERVER in options.monitoring:
            comp.enableComponents.append(messages.MonitoringComponentConfig.EnableComponentsValueListEntryValuesEnum.APISERVER)
        if SCHEDULER in options.monitoring:
            comp.enableComponents.append(messages.MonitoringComponentConfig.EnableComponentsValueListEntryValuesEnum.SCHEDULER)
        if CONTROLLER_MANAGER in options.monitoring:
            comp.enableComponents.append(messages.MonitoringComponentConfig.EnableComponentsValueListEntryValuesEnum.CONTROLLER_MANAGER)
        if STORAGE in options.monitoring:
            comp.enableComponents.append(messages.MonitoringComponentConfig.EnableComponentsValueListEntryValuesEnum.STORAGE)
        if HPA_COMPONENT in options.monitoring:
            comp.enableComponents.append(messages.MonitoringComponentConfig.EnableComponentsValueListEntryValuesEnum.HPA)
        if POD in options.monitoring:
            comp.enableComponents.append(messages.MonitoringComponentConfig.EnableComponentsValueListEntryValuesEnum.POD)
        if DAEMONSET in options.monitoring:
            comp.enableComponents.append(messages.MonitoringComponentConfig.EnableComponentsValueListEntryValuesEnum.DAEMONSET)
        if DEPLOYMENT in options.monitoring:
            comp.enableComponents.append(messages.MonitoringComponentConfig.EnableComponentsValueListEntryValuesEnum.DEPLOYMENT)
        if STATEFULSET in options.monitoring:
            comp.enableComponents.append(messages.MonitoringComponentConfig.EnableComponentsValueListEntryValuesEnum.STATEFULSET)
        if CADVISOR in options.monitoring:
            comp.enableComponents.append(messages.MonitoringComponentConfig.EnableComponentsValueListEntryValuesEnum.CADVISOR)
        if KUBELET in options.monitoring:
            comp.enableComponents.append(messages.MonitoringComponentConfig.EnableComponentsValueListEntryValuesEnum.KUBELET)
        config.componentConfig = comp
    if options.enable_dataplane_v2_metrics:
        adv_obs = messages.AdvancedDatapathObservabilityConfig(enableMetrics=True)
    if options.disable_dataplane_v2_metrics:
        adv_obs = messages.AdvancedDatapathObservabilityConfig(enableMetrics=False)
    if options.dataplane_v2_observability_mode:
        relay_mode = None
        opts_name = options.dataplane_v2_observability_mode.upper()
        if opts_name == 'DISABLED':
            relay_mode = messages.AdvancedDatapathObservabilityConfig.RelayModeValueValuesEnum.DISABLED
        elif opts_name == 'INTERNAL_CLUSTER_SERVICE':
            relay_mode = messages.AdvancedDatapathObservabilityConfig.RelayModeValueValuesEnum.INTERNAL_CLUSTER_SERVICE
        elif opts_name == 'INTERNAL_VPC_LB':
            relay_mode = messages.AdvancedDatapathObservabilityConfig.RelayModeValueValuesEnum.INTERNAL_VPC_LB
        elif opts_name == 'EXTERNAL_LB':
            relay_mode = messages.AdvancedDatapathObservabilityConfig.RelayModeValueValuesEnum.EXTERNAL_LB
        else:
            raise util.Error(DPV2_OBS_ERROR_MSG.format(mode=options.dataplane_v2_observability_mode))
        if adv_obs:
            adv_obs = messages.AdvancedDatapathObservabilityConfig(enableMetrics=adv_obs.enableMetrics, relayMode=relay_mode)
        else:
            adv_obs = messages.AdvancedDatapathObservabilityConfig(relayMode=relay_mode)
    if options.enable_dataplane_v2_flow_observability:
        if adv_obs:
            adv_obs = messages.AdvancedDatapathObservabilityConfig(enableMetrics=adv_obs.enableMetrics, enableRelay=True)
        else:
            adv_obs = messages.AdvancedDatapathObservabilityConfig(enableRelay=True)
    if options.disable_dataplane_v2_flow_observability:
        if adv_obs:
            adv_obs = messages.AdvancedDatapathObservabilityConfig(enableMetrics=adv_obs.enableMetrics, enableRelay=False)
        else:
            adv_obs = messages.AdvancedDatapathObservabilityConfig(enableRelay=False)
    if comp is None and prom is None and (adv_obs is None):
        return None
    if hasattr(config, 'advancedDatapathObservabilityConfig'):
        config.advancedDatapathObservabilityConfig = adv_obs
    return config