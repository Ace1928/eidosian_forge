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
def _AddonsConfig(self, disable_ingress=None, disable_hpa=None, disable_dashboard=None, disable_network_policy=None, enable_node_local_dns=None, enable_gcepd_csi_driver=None, enable_filestore_csi_driver=None, enable_application_manager=None, enable_cloud_build=None, enable_backup_restore=None, enable_gcsfuse_csi_driver=None, enable_stateful_ha=None, enable_parallelstore_csi_driver=None):
    """Generates an AddonsConfig object given specific parameters.

    Args:
      disable_ingress: whether to disable the GCLB ingress controller.
      disable_hpa: whether to disable the horizontal pod autoscaling controller.
      disable_dashboard: whether to disable the Kubernetes Dashboard.
      disable_network_policy: whether to disable NetworkPolicy enforcement.
      enable_node_local_dns: whether to enable NodeLocalDNS cache.
      enable_gcepd_csi_driver: whether to enable GcePersistentDiskCsiDriver.
      enable_filestore_csi_driver: wherher to enable GcpFilestoreCsiDriver.
      enable_application_manager: whether to enable ApplicationManager.
      enable_cloud_build: whether to enable CloudBuild.
      enable_backup_restore: whether to enable BackupRestore.
      enable_gcsfuse_csi_driver: whether to enable GcsFuseCsiDriver.
      enable_stateful_ha: whether to enable StatefulHA addon.
      enable_parallelstore_csi_driver: whether to enable ParallelstoreCsiDriver.

    Returns:
      An AddonsConfig object that contains the options defining what addons to
      run in the cluster.
    """
    addons = self.messages.AddonsConfig()
    if disable_ingress is not None:
        addons.httpLoadBalancing = self.messages.HttpLoadBalancing(disabled=disable_ingress)
    if disable_hpa is not None:
        addons.horizontalPodAutoscaling = self.messages.HorizontalPodAutoscaling(disabled=disable_hpa)
    if disable_dashboard is not None:
        addons.kubernetesDashboard = self.messages.KubernetesDashboard(disabled=disable_dashboard)
    if disable_network_policy is not None:
        addons.networkPolicyConfig = self.messages.NetworkPolicyConfig(disabled=disable_network_policy)
    if enable_node_local_dns is not None:
        addons.dnsCacheConfig = self.messages.DnsCacheConfig(enabled=enable_node_local_dns)
    if enable_gcepd_csi_driver:
        addons.gcePersistentDiskCsiDriverConfig = self.messages.GcePersistentDiskCsiDriverConfig(enabled=True)
    if enable_filestore_csi_driver:
        addons.gcpFilestoreCsiDriverConfig = self.messages.GcpFilestoreCsiDriverConfig(enabled=True)
    if enable_application_manager:
        addons.kalmConfig = self.messages.KalmConfig(enabled=True)
    if enable_cloud_build:
        addons.cloudBuildConfig = self.messages.CloudBuildConfig(enabled=True)
    if enable_backup_restore:
        addons.gkeBackupAgentConfig = self.messages.GkeBackupAgentConfig(enabled=True)
    if enable_gcsfuse_csi_driver:
        addons.gcsFuseCsiDriverConfig = self.messages.GcsFuseCsiDriverConfig(enabled=True)
    if enable_stateful_ha:
        addons.statefulHaConfig = self.messages.StatefulHAConfig(enabled=True)
    if enable_parallelstore_csi_driver:
        addons.parallelstoreCsiDriverConfig = self.messages.ParallelstoreCsiDriverConfig(enabled=True)
    return addons