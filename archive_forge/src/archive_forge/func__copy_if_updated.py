import copy
from oslo_serialization import jsonutils
from urllib import parse
from saharaclient._i18n import _
def _copy_if_updated(self, data, **kwargs):
    for var_name, var_value in kwargs.items():
        if not isinstance(var_value, NotUpdated):
            data[var_name] = var_value