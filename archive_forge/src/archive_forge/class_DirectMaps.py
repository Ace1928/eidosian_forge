import ast
import copy
import re
import flask
import jsonschema
from oslo_config import cfg
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
class DirectMaps(object):
    """An abstraction around the remote matches.

    Each match is treated internally as a list.
    """

    def __init__(self):
        self._matches = []

    def __str__(self):
        """Return the direct map array as a string."""
        return '%s' % self._matches

    def add(self, values):
        """Add a matched value to the list of matches.

        :param list value: the match to save

        """
        self._matches.append(values)

    def __getitem__(self, idx):
        """Used by Python when executing ``''.format(*DirectMaps())``."""
        value = self._matches[idx]
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        else:
            return value