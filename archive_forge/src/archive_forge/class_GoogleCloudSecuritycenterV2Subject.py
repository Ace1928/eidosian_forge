from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Subject(_messages.Message):
    """Represents a Kubernetes subject.

  Enums:
    KindValueValuesEnum: Authentication type for the subject.

  Fields:
    kind: Authentication type for the subject.
    name: Name for the subject.
    ns: Namespace for the subject.
  """

    class KindValueValuesEnum(_messages.Enum):
        """Authentication type for the subject.

    Values:
      AUTH_TYPE_UNSPECIFIED: Authentication is not specified.
      USER: User with valid certificate.
      SERVICEACCOUNT: Users managed by Kubernetes API with credentials stored
        as secrets.
      GROUP: Collection of users.
    """
        AUTH_TYPE_UNSPECIFIED = 0
        USER = 1
        SERVICEACCOUNT = 2
        GROUP = 3
    kind = _messages.EnumField('KindValueValuesEnum', 1)
    name = _messages.StringField(2)
    ns = _messages.StringField(3)