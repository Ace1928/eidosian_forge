import collections
import functools
import weakref
from heat.common import exception
from heat.common.i18n import _
from heat.engine import conditions
from heat.engine import function
from heat.engine import output
from heat.engine import template
@classmethod
def _parse_resource_field(cls, key, valid_types, typename, rsrc_name, rsrc_data, parse_func):
    """Parse a field in a resource definition.

        :param key: The name of the key
        :param valid_types: Valid types for the parsed output
        :param typename: Description of valid type to include in error output
        :param rsrc_name: The resource name
        :param rsrc_data: The unparsed resource definition data
        :param parse_func: A function to parse the data, which takes the
            contents of the field and its path in the template as arguments.
        """
    if key in rsrc_data:
        data = parse_func(rsrc_data[key], '.'.join([cls.RESOURCES, rsrc_name, key]))
        if not isinstance(data, valid_types):
            args = {'name': rsrc_name, 'key': key, 'typename': typename}
            message = _('Resource %(name)s %(key)s type must be %(typename)s') % args
            raise TypeError(message)
        return data
    else:
        return None