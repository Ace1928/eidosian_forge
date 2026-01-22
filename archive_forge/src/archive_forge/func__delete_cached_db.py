import atexit
import os.path
import shutil
import tempfile
import fixtures
import glance_store
from oslo_config import cfg
from oslo_db import options
import glance.common.client
from glance.common import config
import glance.db.sqlalchemy.api
from glance import tests as glance_tests
from glance.tests import utils as test_utils
def _delete_cached_db():
    try:
        os.remove(os.environ[glance_db_env])
    except Exception:
        glance_tests.logger.exception('Error cleaning up the file %s' % os.environ[glance_db_env])