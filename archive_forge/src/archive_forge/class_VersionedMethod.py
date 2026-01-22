import functools
import logging
import os
import pkgutil
import re
import traceback
from oslo_utils import strutils
from zunclient import exceptions
from zunclient.i18n import _
class VersionedMethod(object):

    def __init__(self, name, start_version, end_version, func):
        """Versioning information for a single method

        :param name: Name of the method
        :param start_version: Minimum acceptable version
        :param end_version: Maximum acceptable_version
        :param func: Method to call

        Minimum and maximums are inclusive
        """
        self.name = name
        self.start_version = start_version
        self.end_version = end_version
        self.func = func

    def __str__(self):
        return 'Version Method %s: min: %s, max: %s' % (self.name, self.start_version, self.end_version)

    def __repr__(self):
        return '<VersionedMethod %s>' % self.name