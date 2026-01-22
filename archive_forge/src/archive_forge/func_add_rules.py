from collections import abc
from oslo_config import cfg
from oslo_log import log as logging
from oslo_policy import opts
from oslo_policy import policy
from glance.common import exception
from glance.domain import proxy
from glance import policies
def add_rules(self, rules):
    """Add new rules to the Rules object"""
    self.set_rules(rules, overwrite=False, use_conf=self.use_conf)