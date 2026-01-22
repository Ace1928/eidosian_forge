from oslo_versionedobjects import fields as obj_fields
from neutron_lib._i18n import _
from neutron_lib.services.logapi import constants as log_const
class SecurityEventField(obj_fields.AutoTypedField):
    AUTO_TYPE = SecurityEvent(valid_values=log_const.LOG_EVENTS)