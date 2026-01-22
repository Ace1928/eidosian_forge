from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.assured import util
from googlecloudsdk.calliope.base import ReleaseTrack
def CreateAssuredWorkload(display_name=None, compliance_regime=None, partner=None, partner_permissions=None, billing_account=None, next_rotation_time=None, rotation_period=None, labels=None, etag=None, provisioned_resources_parent=None, resource_settings=None, enable_sovereign_controls=None, violation_notifications_enabled=None, release_track=ReleaseTrack.GA):
    """Construct an Assured Workload message for Assured Workloads Beta API requests.

  Args:
    display_name: str, display name of the Assured Workloads environment.
    compliance_regime: str, the compliance regime, which is one of:
      FEDRAMP_MODERATE, FEDRAMP_HIGH, IL4 or CJIS.
    partner: str, the partner regime/controls.
    partner_permissions: dict, dictionary of permission names and values for the
      partner regime.
    billing_account: str, the billing account of the Assured Workloads
      environment in the form: billingAccounts/{BILLING_ACCOUNT_ID}
    next_rotation_time: str, the next key rotation time for the Assured
      Workloads environment, for example: 2020-12-30T10:15:00.00Z
    rotation_period: str, the time between key rotations, for example: 172800s.
    labels: dict, dictionary of label keys and values of the Assured Workloads
      environment.
    etag: str, the etag of the Assured Workloads environment.
    provisioned_resources_parent: str, parent of provisioned projects, e.g.
      folders/{FOLDER_ID}.
    resource_settings: list of key=value pairs to set customized resource
      settings, which can be one of the following: consumer-project-id,
      consumer-project-name, encryption-keys-project-id,
      encryption-keys-project-name or keyring-id, for example:
      consumer-project-id={ID1},encryption-keys-project-id={ID2}
    enable_sovereign_controls: bool, whether to enable sovereign controls for
      the Assured Workloads environment.
    violation_notifications_enabled: bool, whether email notifications are
      enabled or disabled
    release_track: ReleaseTrack, gcloud release track being used

  Returns:
    A populated Assured Workloads message for the Assured Workloads Beta API.
  """
    workload_message = GetWorkloadMessage(release_track)
    workload = workload_message()
    if etag:
        workload.etag = etag
    if billing_account:
        workload.billingAccount = billing_account
    if display_name:
        workload.displayName = display_name
    if violation_notifications_enabled:
        workload.violationNotificationsEnabled = GetViolationNotificationsEnabled(violation_notifications_enabled)
    if labels:
        workload.labels = CreateLabels(labels, workload_message)
    if compliance_regime:
        workload.complianceRegime = workload_message.ComplianceRegimeValueValuesEnum(compliance_regime)
    if partner:
        workload.partner = workload_message.PartnerValueValuesEnum(partner)
    if partner_permissions:
        workload.partnerPermissions = GetPartnerPermissions(release_track)(dataLogsViewer=partner_permissions['data-logs-viewer'])
    if provisioned_resources_parent:
        workload.provisionedResourcesParent = provisioned_resources_parent
    if next_rotation_time and rotation_period:
        workload.kmsSettings = GetKmsSettings(release_track)(nextRotationTime=next_rotation_time, rotationPeriod=rotation_period)
    if resource_settings:
        workload.resourceSettings = CreateResourceSettingsList(resource_settings, release_track)
    if enable_sovereign_controls:
        workload.enableSovereignControls = enable_sovereign_controls
    return workload