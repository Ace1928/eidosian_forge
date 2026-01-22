import ast
import copy
import re
import flask
import jsonschema
from oslo_config import cfg
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _verify_all_requirements(self, requirements, assertion):
    """Compare remote requirements of a rule against the assertion.

        If a value of ``None`` is returned, the rule with this assertion
        doesn't apply.
        If an array of zero length is returned, then there are no direct
        mappings to be performed, but the rule is valid.
        Otherwise, then it will first attempt to filter the values according
        to blacklist or whitelist rules and finally return the values in
        order, to be directly mapped.

        :param requirements: list of remote requirements from rules
        :type requirements: list

        Example requirements::

            [
                {
                    "type": "UserName"
                },
                {
                    "type": "orgPersonType",
                    "any_one_of": [
                        "Customer"
                    ]
                },
                {
                    "type": "ADFS_GROUPS",
                    "whitelist": [
                        "g1", "g2", "g3", "g4"
                    ]
                }
            ]

        :param assertion: dict of attributes from an IdP
        :type assertion: dict

        Example assertion::

            {
                'UserName': ['testacct'],
                'LastName': ['Account'],
                'orgPersonType': ['Tester'],
                'Email': ['testacct@example.com'],
                'FirstName': ['Test'],
                'ADFS_GROUPS': ['g1', 'g2']
            }

        :returns: identity values used to update local
        :rtype: keystone.federation.utils.DirectMaps or None

        """
    direct_maps = DirectMaps()
    for requirement in requirements:
        requirement_type = requirement['type']
        direct_map_values = assertion.get(requirement_type)
        regex = requirement.get('regex', False)
        if not direct_map_values:
            return None
        any_one_values = requirement.get(self._EvalType.ANY_ONE_OF)
        if any_one_values is not None:
            if self._evaluate_requirement(any_one_values, direct_map_values, self._EvalType.ANY_ONE_OF, regex):
                continue
            else:
                return None
        not_any_values = requirement.get(self._EvalType.NOT_ANY_OF)
        if not_any_values is not None:
            if self._evaluate_requirement(not_any_values, direct_map_values, self._EvalType.NOT_ANY_OF, regex):
                continue
            else:
                return None
        blacklisted_values = requirement.get(self._EvalType.BLACKLIST)
        whitelisted_values = requirement.get(self._EvalType.WHITELIST)
        if blacklisted_values is not None:
            direct_map_values = self._evaluate_requirement(blacklisted_values, direct_map_values, self._EvalType.BLACKLIST, regex)
        elif whitelisted_values is not None:
            direct_map_values = self._evaluate_requirement(whitelisted_values, direct_map_values, self._EvalType.WHITELIST, regex)
        direct_maps.add(direct_map_values)
        LOG.debug('updating a direct mapping: %s', direct_map_values)
    return direct_maps