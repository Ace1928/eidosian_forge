import collections.abc
import copy
import logging
import os
import typing as ty
import warnings
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import strutils
import yaml
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy._i18n import _
from oslo_policy import _parser
from oslo_policy import opts
def _cycle_check(self, check, seen=None):
    """Check if RuleChecks cycle.

        Looking for something like::

            "foo": "rule:bar"
            "bar": "rule:foo"

        :param check: The check to search for.
        :param seen: A set of previously seen rules, else None.
        """
    if seen is None:
        seen = set()
    if isinstance(check, RuleCheck):
        if check.match in seen:
            return True
        seen.add(check.match)
        if check.match in self.rules:
            if self._cycle_check(self.rules[check.match], seen):
                return True
    rules = getattr(check, 'rules', None)
    if rules:
        for rule in rules:
            if self._cycle_check(rule, seen.copy()):
                return True
    return False