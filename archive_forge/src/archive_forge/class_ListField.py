from oslo_serialization import jsonutils as json
from oslo_versionedobjects import fields
class ListField(fields.AutoTypedField):
    AUTO_TYPE = fields.List(fields.FieldType())