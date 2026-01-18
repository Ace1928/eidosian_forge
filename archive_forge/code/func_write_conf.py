import abc
import atexit
import datetime
import errno
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from testtools import content as ttc
import textwrap
import time
from unittest import mock
import urllib.parse as urlparse
import uuid
import fixtures
import glance_store
from os_win import utilsfactory as os_win_utilsfactory
from oslo_config import cfg
from oslo_serialization import jsonutils
import testtools
import webob
from glance.common import config
from glance.common import utils
from glance.common import wsgi
from glance.db.sqlalchemy import api as db_api
from glance import tests as glance_tests
from glance.tests import utils as test_utils
import glance.async_
def write_conf(self, **kwargs):
    """
        Writes the configuration file for the server to its intended
        destination.  Returns the name of the configuration file and
        the over-ridden config content (may be useful for populating
        error messages).
        """
    if not self.conf_base:
        raise RuntimeError('Subclass did not populate config_base!')
    conf_override = self.__dict__.copy()
    if kwargs:
        conf_override.update(**kwargs)
    conf_dir = os.path.join(self.test_dir, 'etc')
    conf_filepath = os.path.join(conf_dir, '%s.conf' % self.server_name)
    if os.path.exists(conf_filepath):
        os.unlink(conf_filepath)
    paste_conf_filepath = conf_filepath.replace('.conf', '-paste.ini')
    if os.path.exists(paste_conf_filepath):
        os.unlink(paste_conf_filepath)
    utils.safe_mkdirs(conf_dir)

    def override_conf(filepath, overridden):
        with open(filepath, 'w') as conf_file:
            conf_file.write(overridden)
            conf_file.flush()
            return conf_file.name
    overridden_core = self.conf_base % conf_override
    self.conf_file_name = override_conf(conf_filepath, overridden_core)
    overridden_paste = ''
    if self.paste_conf_base:
        overridden_paste = self.paste_conf_base % conf_override
        override_conf(paste_conf_filepath, overridden_paste)
    overridden = '==Core config==\n%s\n==Paste config==\n%s' % (overridden_core, overridden_paste)
    return (self.conf_file_name, overridden)