import functools
import flask
from oslo_log import log
from oslo_policy import opts
from oslo_policy import policy as common_policy
from oslo_utils import strutils
from keystone.common import authorization
from keystone.common import context
from keystone.common import policies
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _name_is_changing(rule):
    deprecated_rule = rule.deprecated_rule
    return deprecated_rule and deprecated_rule.name != rule.name and (deprecated_rule.name in self._enforcer.file_rules)