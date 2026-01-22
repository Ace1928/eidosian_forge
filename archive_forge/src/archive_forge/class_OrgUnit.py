from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrgUnit(_messages.Message):
    """JSON template for Org Unit resource in Directory API.

  Fields:
    blockInheritance: Should block inheritance
    description: Description of OrgUnit
    etag: ETag of the resource.
    kind: Kind of resource this is.
    name: Name of OrgUnit
    orgUnitId: Id of OrgUnit
    orgUnitPath: Path of OrgUnit
    parentOrgUnitId: Id of parent OrgUnit
    parentOrgUnitPath: Path of parent OrgUnit
  """
    blockInheritance = _messages.BooleanField(1)
    description = _messages.StringField(2)
    etag = _messages.StringField(3)
    kind = _messages.StringField(4, default=u'admin#directory#orgUnit')
    name = _messages.StringField(5)
    orgUnitId = _messages.StringField(6)
    orgUnitPath = _messages.StringField(7)
    parentOrgUnitId = _messages.StringField(8)
    parentOrgUnitPath = _messages.StringField(9)