from __future__ import (absolute_import, division, print_function)
import ansible.plugins.loader as plugin_loader
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleLookupError, AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible.plugins.lookup import LookupBase
from ansible.utils.sentinel import Sentinel
def _get_plugin_config(pname, ptype, config, variables):
    try:
        loader = getattr(plugin_loader, '%s_loader' % ptype)
        p = loader.get(pname, class_only=True)
        if p is None:
            raise AnsibleLookupError('Unable to load %s plugin "%s"' % (ptype, pname))
        result, origin = C.config.get_config_value_and_origin(config, plugin_type=ptype, plugin_name=p._load_name, variables=variables)
    except AnsibleLookupError:
        raise
    except AnsibleError as e:
        msg = to_native(e)
        if 'was not defined' in msg:
            raise MissingSetting(msg, orig_exc=e)
        raise e
    return (result, origin)