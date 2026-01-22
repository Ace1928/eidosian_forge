import copy
from oslo_serialization import jsonutils
from urllib import parse
from saharaclient._i18n import _
def _plurify_resource_name(self):
    return self.resource_class.resource_name + 's'