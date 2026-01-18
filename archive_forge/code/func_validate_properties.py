from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
@staticmethod
def validate_properties(properties):
    """Validate properties for the resource.

        Validates to ensure nothing in value_specs overwrites any key that
        exists in the schema.

        Also ensures that shared and tenant_id is not specified
        in value_specs.
        """
    if 'value_specs' in properties:
        banned_keys = set(['shared', 'tenant_id']).union(set(properties))
        found = banned_keys.intersection(set(properties['value_specs']))
        if found:
            return '%s not allowed in value_specs' % ', '.join(found)