import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState

        Update an API access rules.
        The new rule you specify fully replaces the old rule. Therefore,
        for a parameter that is not specified, any previously set value
        is deleted.

        :param      api_access_rule_id: The id of the rule we want to update
        (required).
        :type       api_access_rule_id: ``str``

        :param      ca_ids: One or more IDs of Client Certificate Authorities
        (CAs).
        :type       ca_ids: ``List`` of ``str``

        :param      cns: One or more Client Certificate Common Names (CNs).
        If this parameter is specified, you must also specify the ca_ids
        parameter.
        :type       cns: ``List`` of ``str``

        :param      description: The description of the new rule.
        :type       description: ``str``

        :param      ip_ranges: One or more IP ranges, in CIDR notation
        (for example, 192.0.2.0/16).
        :type       ip_ranges: ``List`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: a List of API access rules.
        :rtype: ``List`` of ``dict`` if successful or  ``dict``
        