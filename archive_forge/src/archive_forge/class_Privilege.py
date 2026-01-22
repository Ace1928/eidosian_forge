from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Privilege(_messages.Message):
    """JSON template for privilege resource in Directory API.

  Fields:
    childPrivileges: A list of child privileges. Privileges for a service form
      a tree. Each privilege can have a list of child privileges; this list is
      empty for a leaf privilege.
    etag: ETag of the resource.
    isOuScopable: If the privilege can be restricted to an organization unit.
    kind: The type of the API resource. This is always
      admin#directory#privilege.
    privilegeName: The name of the privilege.
    serviceId: The obfuscated ID of the service this privilege is for. This
      value is returned with Privileges.list().
    serviceName: The name of the service this privilege is for.
  """
    childPrivileges = _messages.MessageField('Privilege', 1, repeated=True)
    etag = _messages.StringField(2)
    isOuScopable = _messages.BooleanField(3)
    kind = _messages.StringField(4, default=u'admin#directory#privilege')
    privilegeName = _messages.StringField(5)
    serviceId = _messages.StringField(6)
    serviceName = _messages.StringField(7)