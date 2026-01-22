import copy
from oslo_serialization import jsonutils
from urllib import parse
from saharaclient._i18n import _
def _copy_if_defined(self, data, **kwargs):
    for var_name, var_value in kwargs.items():
        if var_value is not None:
            data[var_name] = var_value