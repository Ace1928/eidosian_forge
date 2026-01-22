import collections
import copy
import enum
import sys
from typing import List
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import http_retry
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
def _GetServices(policy: str, policy_name: str, force: bool, validate_only: bool):
    """Get list of services from operation response."""
    operation = UpdateConsumerPolicyV2Alpha(policy, policy_name, force, validate_only)
    services = set()
    if operation.response:
        reposonse_dict = encoding.MessageToPyValue(operation.response)
        if 'enableRules' in reposonse_dict.keys():
            enable_rules = reposonse_dict['enableRules']
            keys = list(set().union(*(d.keys() for d in enable_rules)))
            if 'services' in keys:
                services_enabled = enable_rules[keys.index('services')]
                for service in services_enabled['services']:
                    services.add(service)
        log.status.Print("Consumer policy '" + policy_name + "' (validate-only):")
        for service in services:
            log.status.Print(service)