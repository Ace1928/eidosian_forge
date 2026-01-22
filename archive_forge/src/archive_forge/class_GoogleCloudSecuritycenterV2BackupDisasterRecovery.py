from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2BackupDisasterRecovery(_messages.Message):
    """Information related to Google Cloud Backup and DR Service findings.

  Fields:
    appliance: The name of the Backup and DR appliance that captures, moves,
      and manages the lifecycle of backup data. For example, `backup-
      server-57137`.
    applications: The names of Backup and DR applications. An application is a
      VM, database, or file system on a managed host monitored by a backup and
      recovery appliance. For example, `centos7-01-vol00`, `centos7-01-vol01`,
      `centos7-01-vol02`.
    backupCreateTime: The timestamp at which the Backup and DR backup was
      created.
    backupTemplate: The name of a Backup and DR template which comprises one
      or more backup policies. See the [Backup and DR
      documentation](https://cloud.google.com/backup-disaster-
      recovery/docs/concepts/backup-plan#temp) for more information. For
      example, `snap-ov`.
    backupType: The backup type of the Backup and DR image. For example,
      `Snapshot`, `Remote Snapshot`, `OnVault`.
    host: The name of a Backup and DR host, which is managed by the backup and
      recovery appliance and known to the management console. The host can be
      of type Generic (for example, Compute Engine, SQL Server, Oracle DB, SMB
      file system, etc.), vCenter, or an ESX server. See the [Backup and DR
      documentation on hosts](https://cloud.google.com/backup-disaster-
      recovery/docs/configuration/manage-hosts-and-their-applications) for
      more information. For example, `centos7-01`.
    policies: The names of Backup and DR policies that are associated with a
      template and that define when to run a backup, how frequently to run a
      backup, and how long to retain the backup image. For example,
      `onvaults`.
    policyOptions: The names of Backup and DR advanced policy options of a
      policy applying to an application. See the [Backup and DR documentation
      on policy options](https://cloud.google.com/backup-disaster-
      recovery/docs/create-plan/policy-settings). For example,
      `skipofflineappsincongrp, nounmap`.
    profile: The name of the Backup and DR resource profile that specifies the
      storage media for backups of application and VM data. See the [Backup
      and DR documentation on profiles](https://cloud.google.com/backup-
      disaster-recovery/docs/concepts/backup-plan#profile). For example,
      `GCP`.
    storagePool: The name of the Backup and DR storage pool that the backup
      and recovery appliance is storing data in. The storage pool could be of
      type Cloud, Primary, Snapshot, or OnVault. See the [Backup and DR
      documentation on storage pools](https://cloud.google.com/backup-
      disaster-recovery/docs/concepts/storage-pools). For example,
      `DiskPoolOne`.
  """
    appliance = _messages.StringField(1)
    applications = _messages.StringField(2, repeated=True)
    backupCreateTime = _messages.StringField(3)
    backupTemplate = _messages.StringField(4)
    backupType = _messages.StringField(5)
    host = _messages.StringField(6)
    policies = _messages.StringField(7, repeated=True)
    policyOptions = _messages.StringField(8, repeated=True)
    profile = _messages.StringField(9)
    storagePool = _messages.StringField(10)