import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
class TreeRule(BaseRule):
    """A tree rule is non-terminal meaning it will never be returned to a provider.
    Additionally this means it has no attributes that need to be resolved.
    """

    def __init__(self, rules, **kwargs):
        super().__init__(**kwargs)
        self.rules = [RuleCreator.create(**rule) for rule in rules]

    def evaluate(self, scope_vars, rule_lib):
        """If a tree rule's conditions are met, iterate its sub-rules
        and return first result found.

        :type scope_vars: dict
        :type rule_lib: RuleSetStandardLibrary
        :rtype: RuleSetEndpoint/EndpointResolutionError
        """
        if self.evaluate_conditions(scope_vars, rule_lib):
            for rule in self.rules:
                rule_result = rule.evaluate(scope_vars.copy(), rule_lib)
                if rule_result:
                    return rule_result
        return None