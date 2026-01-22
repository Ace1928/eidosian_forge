from datetime import datetime
from oslo_versionedobjects import base
from oslo_versionedobjects import fields as obj_fields
@base.VersionedObjectRegistry.register
class IOTLightbulb(base.VersionedObject):
    """Simple light bulb class with some data about it."""
    VERSION = '1.0'
    OBJ_PROJECT_NAMESPACE = 'versionedobjects.examples'
    fields = {'serial': obj_fields.StringField(), 'manufactured_on': obj_fields.DateTimeField()}