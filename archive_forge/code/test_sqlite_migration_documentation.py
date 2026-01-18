from contextlib import contextmanager
import os
import sqlite3
import tempfile
import time
from unittest import mock
import uuid
from oslo_config import cfg
from glance import sqlite_migration
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils

        Returns a context manager that produces a database connection that
        self-closes and calls rollback if an error occurs while using the
        database connection
        