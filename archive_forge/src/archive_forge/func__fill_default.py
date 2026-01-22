from webob import exc
from neutron_lib._i18n import _
from neutron_lib.api.definitions import network
from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import subnet
from neutron_lib.api.definitions import subnetpool
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions
def _fill_default(res_dict, attr_name, attr_spec):
    if attr_spec.get('default_overrides_none'):
        if res_dict.get(attr_name) is None:
            res_dict[attr_name] = attr_spec.get('default')
            return
    res_dict[attr_name] = res_dict.get(attr_name, attr_spec.get('default'))