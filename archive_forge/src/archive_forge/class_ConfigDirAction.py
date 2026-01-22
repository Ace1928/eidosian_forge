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
class ConfigDirAction(argparse.Action):
    """An argparse action for --config-dir.

        As each --config-dir option is encountered, this action sets the
        config_dir attribute on the _Namespace object but also parses the
        configuration files and stores the values found also in the
        _Namespace object.
        """

    def __call__(self, parser, namespace, values, option_string=None):
        """Handle a --config-dir command line argument.

            :raises: ConfigFileParseError, ConfigFileValueError,
                     ConfigDirNotFoundError
            """
        namespace._config_dirs.append(values)
        setattr(namespace, self.dest, values)
        values = os.path.expanduser(values)
        if not os.path.exists(values):
            raise ConfigDirNotFoundError(values)
        config_dir_glob = os.path.join(values, '*.conf')
        for config_file in sorted(glob.glob(config_dir_glob)):
            ConfigParser._parse_file(config_file, namespace)