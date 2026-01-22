from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigManagementConfigSync(_messages.Message):
    """Configuration for Config Sync

  Fields:
    allowVerticalScale: Set to true to allow the vertical scaling. Defaults to
      false which disallows vertical scaling. This field is deprecated.
    enabled: Enables the installation of ConfigSync. If set to true,
      ConfigSync resources will be created and the other ConfigSync fields
      will be applied if exist. If set to false, all other ConfigSync fields
      will be ignored, ConfigSync resources will be deleted. If omitted,
      ConfigSync resources will be managed depends on the presence of the git
      or oci field.
    git: Git repo configuration for the cluster.
    metricsGcpServiceAccountEmail: The Email of the Google Cloud Service
      Account (GSA) used for exporting Config Sync metrics to Cloud Monitoring
      and Cloud Monarch when Workload Identity is enabled. The GSA should have
      the Monitoring Metric Writer (roles/monitoring.metricWriter) IAM role.
      The Kubernetes ServiceAccount `default` in the namespace `config-
      management-monitoring` should be bound to the GSA.
    oci: OCI repo configuration for the cluster
    preventDrift: Set to true to enable the Config Sync admission webhook to
      prevent drifts. If set to `false`, disables the Config Sync admission
      webhook and does not prevent drifts.
    sourceFormat: Specifies whether the Config Sync Repo is in "hierarchical"
      or "unstructured" mode.
    stopSyncing: Set to true to stop syncing configs for a single cluster when
      automatic Feature management is enabled. Default to false. The field
      will be ignored when automatic Feature management is disabled.
  """
    allowVerticalScale = _messages.BooleanField(1)
    enabled = _messages.BooleanField(2)
    git = _messages.MessageField('ConfigManagementGitConfig', 3)
    metricsGcpServiceAccountEmail = _messages.StringField(4)
    oci = _messages.MessageField('ConfigManagementOciConfig', 5)
    preventDrift = _messages.BooleanField(6)
    sourceFormat = _messages.StringField(7)
    stopSyncing = _messages.BooleanField(8)