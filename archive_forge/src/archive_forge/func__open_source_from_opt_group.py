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
def _open_source_from_opt_group(self, group_name):
    if not self._ext_mgr:
        self._ext_mgr = stevedore.ExtensionManager('oslo.config.driver', invoke_on_load=True)
    self.register_opt(StrOpt('driver', choices=self._ext_mgr.names(), help=_SOURCE_DRIVER_OPTION_HELP), group=group_name)
    try:
        driver_name = self[group_name].driver
    except ConfigFileValueError as err:
        LOG.error('could not load configuration from %r. %s', group_name, err.msg)
        return None
    if driver_name is None:
        LOG.error("could not load configuration from %r, no 'driver' is set.", group_name)
        return None
    LOG.info('loading configuration from %r using %r', group_name, driver_name)
    driver = self._ext_mgr[driver_name].obj
    try:
        return driver.open_source_from_opt_group(self, group_name)
    except Exception as err:
        LOG.error('could not load configuration from %r using %s driver: %s', group_name, driver_name, err)
        return None