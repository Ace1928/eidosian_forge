from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PosixAccount(_messages.Message):
    """The POSIX account information associated with a Google account.

  Enums:
    OperatingSystemTypeValueValuesEnum: The operating system type where this
      account applies.

  Fields:
    accountId: Output only. A POSIX account identifier.
    gecos: The GECOS (user information) entry for this account.
    gid: The default group ID.
    homeDirectory: The path to the home directory for this account.
    name: Output only. The canonical resource name.
    operatingSystemType: The operating system type where this account applies.
    primary: Only one POSIX account can be marked as primary.
    shell: The path to the logic shell for this account.
    systemId: System identifier for which account the username or uid applies
      to. By default, the empty value is used.
    uid: The user ID.
    username: The username of the POSIX account.
  """

    class OperatingSystemTypeValueValuesEnum(_messages.Enum):
        """The operating system type where this account applies.

    Values:
      OPERATING_SYSTEM_TYPE_UNSPECIFIED: The operating system type associated
        with the user account information is unspecified.
      LINUX: Linux user account information.
      WINDOWS: Windows user account information.
    """
        OPERATING_SYSTEM_TYPE_UNSPECIFIED = 0
        LINUX = 1
        WINDOWS = 2
    accountId = _messages.StringField(1)
    gecos = _messages.StringField(2)
    gid = _messages.IntegerField(3)
    homeDirectory = _messages.StringField(4)
    name = _messages.StringField(5)
    operatingSystemType = _messages.EnumField('OperatingSystemTypeValueValuesEnum', 6)
    primary = _messages.BooleanField(7)
    shell = _messages.StringField(8)
    systemId = _messages.StringField(9)
    uid = _messages.IntegerField(10)
    username = _messages.StringField(11)