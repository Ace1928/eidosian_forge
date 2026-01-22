import functools
import logging
import re
from oslo_utils import strutils
from cinderclient._i18n import _
from cinderclient import exceptions
from cinderclient import utils
def add_substitution(versioned_method):
    _SUBSTITUTIONS.setdefault(versioned_method.name, [])
    _SUBSTITUTIONS[versioned_method.name].append(versioned_method)