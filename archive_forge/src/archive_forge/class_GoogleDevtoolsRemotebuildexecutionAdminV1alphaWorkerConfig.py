from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildexecutionAdminV1alphaWorkerConfig(_messages.Message):
    """Defines the configuration to be used for creating workers in the worker
  pool.

  Messages:
    LabelsValue: Labels associated with the workers. Label keys and values can
      be no longer than 63 characters, can only contain lowercase letters,
      numeric characters, underscores and dashes. International letters are
      permitted. Label keys must start with a letter. Label values are
      optional. There can not be more than 64 labels per resource.

  Fields:
    accelerator: The accelerator card attached to each VM.
    attachedDisks: Optional. Specifies the disks that will be attached.
    diskSizeGb: Required. Size of the disk attached to the worker, in GB. See
      https://cloud.google.com/compute/docs/disks/
    diskType: Required. Disk Type to use for the worker. See [Storage
      options](https://cloud.google.com/compute/docs/disks/#introduction).
      Currently only `pd-standard` and `pd-ssd` are supported.
    labels: Labels associated with the workers. Label keys and values can be
      no longer than 63 characters, can only contain lowercase letters,
      numeric characters, underscores and dashes. International letters are
      permitted. Label keys must start with a letter. Label values are
      optional. There can not be more than 64 labels per resource.
    machineType: Required. Machine type of the worker, such as
      `e2-standard-2`. See https://cloud.google.com/compute/docs/machine-types
      for a list of supported machine types. Note that `f1-micro` and
      `g1-small` are not yet supported.
    maxConcurrentActions: The maximum number of actions a worker can execute
      concurrently.
    minCpuPlatform: Minimum CPU platform to use when creating the worker. See
      [CPU Platforms](https://cloud.google.com/compute/docs/cpu-platforms).
    networkAccess: Determines the type of network access granted to workers.
      Possible values: - "public": Workers can connect to the public internet.
      - "private": Workers can only connect to Google APIs and services. -
      "restricted-private": Workers can only connect to Google APIs that are
      reachable through `restricted.googleapis.com` (`199.36.153.4/30`).
    reserved: Determines whether the worker is reserved (equivalent to a
      Compute Engine on-demand VM and therefore won't be preempted). See
      [Preemptible VMs](https://cloud.google.com/preemptible-vms/) for more
      details.
    soleTenantNodeType: The node type name to be used for sole-tenant nodes.
    userServiceAccounts: Optional. List of user service accounts. The last
      service account in the list is what the user code will run as. The rest
      of the service accounts constitute the impersonation chain. For example,
      if len(user_service_accounts) == 2 and if the VM's service account is
      RBE's P4SA, then RBE'S P4SA should be granted the Service Account Token
      Creator role on user_service_accounts[0] and user_service_accounts[0]
      should be granted the Service Account Token Creator role on
      user_service_accounts[1].
    vmImage: The name of the image used by each VM.
    zones: Optional. Zones in the region where the pool VMs should be. Leave
      empty for no restrictions.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels associated with the workers. Label keys and values can be no
    longer than 63 characters, can only contain lowercase letters, numeric
    characters, underscores and dashes. International letters are permitted.
    Label keys must start with a letter. Label values are optional. There can
    not be more than 64 labels per resource.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    accelerator = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaAcceleratorConfig', 1)
    attachedDisks = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaDisks', 2)
    diskSizeGb = _messages.IntegerField(3)
    diskType = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    machineType = _messages.StringField(6)
    maxConcurrentActions = _messages.IntegerField(7)
    minCpuPlatform = _messages.StringField(8)
    networkAccess = _messages.StringField(9)
    reserved = _messages.BooleanField(10)
    soleTenantNodeType = _messages.StringField(11)
    userServiceAccounts = _messages.StringField(12, repeated=True)
    vmImage = _messages.StringField(13)
    zones = _messages.StringField(14, repeated=True)