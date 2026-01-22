import functools
import logging
import os
import pkgutil
import re
import traceback
from oslo_utils import strutils
from zunclient import exceptions
from zunclient.i18n import _
def _add_substitution(versioned_method):
    _SUBSTITUTIONS.setdefault(versioned_method.name, [])
    _SUBSTITUTIONS[versioned_method.name].append(versioned_method)