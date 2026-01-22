from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WarningsValueListEntryValuesEnum(_messages.Enum):
    """WarningsValueListEntryValuesEnum enum type.

    Values:
      WARNING_UNSPECIFIED: Default type.
      UNSUPPORTED_ACCELERATOR_TYPE: The GmN uses an accelerator type that's
        unsupported in WbI. It will be migrated without an accelerator. Users
        can attach an accelerator after the migration.
      UNSUPPORTED_OS: The GmN uses an operating system that's unsupported in
        WbI (e.g. Debian 10). It will be replaced with Debian 11 in WbI.
      RESERVED_IP_RANGE: This GmN is configured with reserved IP range, which
        is no longer applicable in WbI.
      GOOGLE_MANAGED_NETWORK: This GmN is configured with a Google managed
        network. Please provide the `network` and `subnet` options for the
        migration.
      POST_STARTUP_SCRIPT: This GmN is configured with a post startup script.
        Please optionally provide the `post_startup_script_option` for the
        migration.
      SINGLE_USER: This GmN is configured with single user mode. Please
        optionally provide the `service_account` option for the migration.
    """
    WARNING_UNSPECIFIED = 0
    UNSUPPORTED_ACCELERATOR_TYPE = 1
    UNSUPPORTED_OS = 2
    RESERVED_IP_RANGE = 3
    GOOGLE_MANAGED_NETWORK = 4
    POST_STARTUP_SCRIPT = 5
    SINGLE_USER = 6