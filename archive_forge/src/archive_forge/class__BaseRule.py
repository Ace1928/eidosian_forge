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
class _BaseRule:

    def __init__(self, name, check_str):
        self._name = name
        self._check_str = check_str
        self._check = _parser.parse_rule(self.check_str)

    @property
    def name(self):
        return self._name

    @property
    def check_str(self):
        return self._check_str

    @property
    def check(self):
        return self._check

    def __str__(self):
        return f'"{self.name}": "{self.check_str}"'