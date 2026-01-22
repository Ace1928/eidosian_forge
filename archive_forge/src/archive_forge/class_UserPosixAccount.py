from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserPosixAccount(_messages.Message):
    """JSON template for a POSIX account entry.

  Description of the field
  family: go/fbs-posix.

  Fields:
    accountId: A POSIX account field identifier.
    gecos: The GECOS (user information) for this account.
    gid: The default group ID.
    homeDirectory: The path to the home directory for this account.
    operatingSystemType: The operating system type for this account.
    primary: If this is user's primary account within the SystemId.
    shell: The path to the login shell for this account.
    systemId: System identifier for which account Username or Uid apply to.
    uid: The POSIX compliant user ID.
    username: The username of the account.
  """
    accountId = _messages.StringField(1)
    gecos = _messages.StringField(2)
    gid = _messages.IntegerField(3, variant=_messages.Variant.UINT64)
    homeDirectory = _messages.StringField(4)
    operatingSystemType = _messages.StringField(5)
    primary = _messages.BooleanField(6)
    shell = _messages.StringField(7)
    systemId = _messages.StringField(8)
    uid = _messages.IntegerField(9, variant=_messages.Variant.UINT64)
    username = _messages.StringField(10)