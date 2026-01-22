import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
class EndpointProvider:
    """Derives endpoints from a RuleSet for given input parameters."""

    def __init__(self, ruleset_data, partition_data):
        self.ruleset = RuleSet(**ruleset_data, partitions=partition_data)

    @lru_cache_weakref(maxsize=CACHE_SIZE)
    def resolve_endpoint(self, **input_parameters):
        """Match input parameters to a rule.

        :type input_parameters: dict
        :rtype: RuleSetEndpoint
        """
        params_for_error = input_parameters.copy()
        endpoint = self.ruleset.evaluate(input_parameters)
        if endpoint is None:
            param_string = '\n'.join([f'{key}: {value}' for key, value in params_for_error.items()])
            raise EndpointResolutionError(msg=f'No endpoint found for parameters:\n{param_string}')
        return endpoint