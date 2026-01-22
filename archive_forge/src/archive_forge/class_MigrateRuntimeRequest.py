from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MigrateRuntimeRequest(_messages.Message):
    """Request for migrating a Runtime to a Workbench Instance.

  Enums:
    PostStartupScriptOptionValueValuesEnum: Optional. Specifies the behavior
      of post startup script during migration.

  Fields:
    network: Optional. Name of the VPC that the new Instance is in. This is
      required if the Runtime uses google-managed network. If the Runtime uses
      customer-owned network, it will reuse the same VPC, and this field must
      be empty. Format: `projects/{project_id}/global/networks/{network_id}`
    postStartupScriptOption: Optional. Specifies the behavior of post startup
      script during migration.
    requestId: Optional. Idempotent request UUID.
    serviceAccount: Optional. The service account to be included in the
      Compute Engine instance of the new Workbench Instance when the Runtime
      uses "single user only" mode for permission. If not specified, the
      [Compute Engine default service
      account](https://cloud.google.com/compute/docs/access/service-
      accounts#default_service_account) is used. When the Runtime uses service
      account mode for permission, it will reuse the same service account, and
      this field must be empty.
    subnet: Optional. Name of the subnet that the new Instance is in. This is
      required if the Runtime uses google-managed network. If the Runtime uses
      customer-owned network, it will reuse the same subnet, and this field
      must be empty. Format:
      `projects/{project_id}/regions/{region}/subnetworks/{subnetwork_id}`
  """

    class PostStartupScriptOptionValueValuesEnum(_messages.Enum):
        """Optional. Specifies the behavior of post startup script during
    migration.

    Values:
      POST_STARTUP_SCRIPT_OPTION_UNSPECIFIED: Post startup script option is
        not specified. Default is POST_STARTUP_SCRIPT_OPTION_SKIP.
      POST_STARTUP_SCRIPT_OPTION_SKIP: Not migrate the post startup script to
        the new Workbench Instance.
      POST_STARTUP_SCRIPT_OPTION_RERUN: Redownload and rerun the same post
        startup script as the Google-Managed Notebook.
    """
        POST_STARTUP_SCRIPT_OPTION_UNSPECIFIED = 0
        POST_STARTUP_SCRIPT_OPTION_SKIP = 1
        POST_STARTUP_SCRIPT_OPTION_RERUN = 2
    network = _messages.StringField(1)
    postStartupScriptOption = _messages.EnumField('PostStartupScriptOptionValueValuesEnum', 2)
    requestId = _messages.StringField(3)
    serviceAccount = _messages.StringField(4)
    subnet = _messages.StringField(5)