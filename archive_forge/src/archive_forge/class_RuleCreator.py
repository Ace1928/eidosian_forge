import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
class RuleCreator:
    endpoint = EndpointRule
    error = ErrorRule
    tree = TreeRule

    @classmethod
    def create(cls, **kwargs):
        """Create a rule instance from metadata.

        :rtype: TreeRule/EndpointRule/ErrorRule
        """
        rule_type = kwargs.pop('type')
        try:
            rule_class = getattr(cls, rule_type)
        except AttributeError:
            raise EndpointResolutionError(msg=f'Unknown rule type: {rule_type}. A rule must be of type tree, endpoint or error.')
        else:
            return rule_class(**kwargs)