import argparse
import collections
from collections import abc
import copy
import enum
import errno
import functools
import glob
import inspect
import itertools
import logging
import os
import string
import sys
from oslo_config import iniparser
from oslo_config import sources
import oslo_config.sources._environment as _environment
from oslo_config import types
import stevedore
class StrSubWrapper:
    """Helper class.

        Exposes opt values as a dict for string substitution.
        """

    def __init__(self, conf, group=None, namespace=None):
        """Construct a StrSubWrapper object.

            :param conf: a ConfigOpts object
            :param group: an OptGroup object
            :param namespace: the namespace object that retrieves the option
                              value from
            """
        self.conf = conf
        self.group = group
        self.namespace = namespace

    def __getitem__(self, key):
        """Look up an opt value from the ConfigOpts object.

            :param key: an opt name
            :returns: an opt value
            """
        try:
            group_name, option = key.split('.', 1)
        except ValueError:
            group = self.group
            option = key
        else:
            group = OptGroup(name=group_name)
        try:
            value = self.conf._get(option, group=group, namespace=self.namespace)
        except NoSuchOptError:
            value = self.conf._get(key, namespace=self.namespace)
        if isinstance(value, self.conf.GroupAttr):
            raise TemplateSubstitutionError('substituting group %s not supported' % key)
        if value is None:
            return ''
        return value