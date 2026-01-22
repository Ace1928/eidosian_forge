from functools import partial
from oslo_log import log
import stevedore
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import resource_options as ro
@staticmethod
def _parse_rule_structure(rules, user_id):
    """Validate and parse the rule data structure.

        Rule sets must be in the form of list of lists. The lists may not
        have duplicates and must not be empty. The top-level list may be empty
        indicating that no rules exist.

        :param rules: The list of rules from the user_ref
        :type rules: list
        :param user_id: the user_id, used for logging purposes
        :type user_id: str
        :returns: list of list, duplicates are stripped
        """
    rule_set = []
    if not isinstance(rules, list):
        LOG.error('Corrupt rule data structure for user %(user_id)s, no rules loaded.', {'user_id': user_id})
        return rule_set
    elif not rules:
        return rule_set
    for r_list in rules:
        if not isinstance(r_list, list):
            LOG.info('Ignoring Rule %(type)r; rule must be a list of strings.', {'type': type(r_list)})
            continue
        if r_list:
            _ok_rule = True
            for item in r_list:
                if not isinstance(item, str):
                    LOG.info('Ignoring Rule %(rule)r; rule contains non-string values.', {'rule': r_list})
                    _ok_rule = False
                    break
            if _ok_rule:
                r_list = list(set(r_list))
                rule_set.append(r_list)
    return rule_set