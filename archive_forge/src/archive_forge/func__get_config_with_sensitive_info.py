from oslo_log import log
from keystone import assignment
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.resource.backends import base
from keystone.token import provider as token_provider
def _get_config_with_sensitive_info(self, domain_id, group=None, option=None):
    """Get config for a domain/group/option with sensitive info included.

        This is only used by the methods within this class, which may need to
        check individual groups or options.

        """
    whitelisted = self.list_config_options(domain_id, group, option)
    sensitive = self.list_config_options(domain_id, group, option, sensitive=True)
    sensitive_dict = {s['option']: s['value'] for s in sensitive}
    for each_whitelisted in whitelisted:
        if not isinstance(each_whitelisted['value'], str):
            continue
        original_value = each_whitelisted['value']
        warning_msg = ''
        try:
            each_whitelisted['value'] = each_whitelisted['value'] % sensitive_dict
        except KeyError:
            warning_msg = 'Found what looks like an unmatched config option substitution reference - domain: %(domain)s, group: %(group)s, option: %(option)s, value: %(value)s. Perhaps the config option to which it refers has yet to be added?'
        except (ValueError, TypeError):
            warning_msg = 'Found what looks like an incorrectly constructed config option substitution reference - domain: %(domain)s, group: %(group)s, option: %(option)s, value: %(value)s.'
        if warning_msg:
            LOG.warning(warning_msg, {'domain': domain_id, 'group': each_whitelisted['group'], 'option': each_whitelisted['option'], 'value': original_value})
    return self._list_to_config(whitelisted, sensitive)