from oslo_log import log as logging
from oslo_serialization import jsonutils
from requests import exceptions
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.common import template_format
from heat.common import urlfetch
from heat.engine import attributes
from heat.engine import environment
from heat.engine import properties
from heat.engine.resources import stack_resource
from heat.engine import template
from heat.rpc import api as rpc_api
def _validate_against_facade(self, facade_cls):
    facade_schemata = properties.schemata(facade_cls.properties_schema)
    for n, fs in facade_schemata.items():
        if fs.required and n not in self.properties_schema:
            msg = _('Required property %(n)s for facade %(type)s missing in provider') % {'n': n, 'type': self.type()}
            raise exception.StackValidationFailed(message=msg)
        ps = self.properties_schema.get(n)
        if n in self.properties_schema and fs.allowed_param_prop_type() != ps.type:
            msg = _('Property %(n)s type mismatch between facade %(type)s (%(fs_type)s) and provider (%(ps_type)s)') % {'n': n, 'type': self.type(), 'fs_type': fs.type, 'ps_type': ps.type}
            raise exception.StackValidationFailed(message=msg)
    for n, ps in self.properties_schema.items():
        if ps.required and n not in facade_schemata:
            msg = _('Provider requires property %(n)s unknown in facade %(type)s') % {'n': n, 'type': self.type()}
            raise exception.StackValidationFailed(message=msg)
    facade_attrs = facade_cls.attributes_schema.copy()
    facade_attrs.update(facade_cls.base_attributes_schema)
    for attr in facade_attrs:
        if attr not in self.attributes_schema:
            msg = _('Attribute %(attr)s for facade %(type)s missing in provider') % {'attr': attr, 'type': self.type()}
            raise exception.StackValidationFailed(message=msg)