from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils._text import to_native
from ansible.module_utils.six import integer_types, string_types
from jinja2.exceptions import TemplateSyntaxError
def _to_well_known_type(obj):
    """Convert an ansible internal type to a well-known type
    ie AnsibleUnicode => str

    :param obj: the obj to convert
    :type obj: unknown
    """
    return json.loads(json.dumps(obj))