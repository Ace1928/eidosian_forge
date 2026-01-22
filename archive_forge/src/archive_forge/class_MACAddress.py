import netaddr
from sqlalchemy import types
from neutron_lib._i18n import _
class MACAddress(types.TypeDecorator):
    impl = types.String(64)
    cache_ok = True

    def process_result_value(self, value, dialect):
        return netaddr.EUI(value)

    def process_bind_param(self, value, dialect):
        if not isinstance(value, netaddr.EUI):
            raise AttributeError(_("Received type '%(type)s' and value '%(value)s'. Expecting netaddr.EUI type.") % {'type': type(value), 'value': value})
        return str(value)