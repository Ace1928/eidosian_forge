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
def _reset_database(self, conn_string):
    conn_pieces = urlparse.urlparse(conn_string)
    if conn_string.startswith('sqlite'):
        pass
    elif conn_string.startswith('mysql'):
        database = conn_pieces.path.strip('/')
        loc_pieces = conn_pieces.netloc.split('@')
        host = loc_pieces[1]
        auth_pieces = loc_pieces[0].split(':')
        user = auth_pieces[0]
        password = ''
        if len(auth_pieces) > 1:
            if auth_pieces[1].strip():
                password = '-p%s' % auth_pieces[1]
        sql = 'drop database if exists %(database)s; create database %(database)s;' % {'database': database}
        cmd = 'mysql -u%(user)s %(password)s -h%(host)s -e"%(sql)s"' % {'user': user, 'password': password, 'host': host, 'sql': sql}
        exitcode, out, err = execute(cmd)
        self.assertEqual(0, exitcode)