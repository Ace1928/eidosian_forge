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
def _record_file_rules(self, data, overwrite=False):
    """Store a copy of rules loaded from a file.

        It is useful to be able to distinguish between rules loaded from a file
        and those registered by a consuming service. In order to do so we keep
        a record of rules loaded from a file.

        :param data: The raw contents of a policy file.
        :param overwrite: If True clear out previously loaded rules.
        """
    if overwrite:
        self.file_rules = {}
    parsed_file = parse_file_contents(data)
    redundant_file_rules = []
    for name, check_str in parsed_file.items():
        file_rule = RuleDefault(name, check_str)
        self.file_rules[name] = file_rule
        reg_rule = self.registered_rules.get(name)
        if reg_rule and file_rule == reg_rule:
            redundant_file_rules.append(name)
    if redundant_file_rules:
        LOG.warning('Policy Rules %(names)s specified in policy files are the same as the defaults provided by the service. You can remove these rules from policy files which will make maintenance easier. You can detect these redundant rules by ``oslopolicy-list-redundant`` tool also.', {'names': redundant_file_rules})