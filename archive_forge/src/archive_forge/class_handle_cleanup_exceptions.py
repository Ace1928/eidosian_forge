import re
import traceback
from oslo_log import log
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import client
from manilaclient.tests.functional import utils
class handle_cleanup_exceptions(object):
    """Handle exceptions raised with cleanup operations.

    Always suppress errors when lib_exc.NotFound or lib_exc.Forbidden
    are raised.
    Suppress all other exceptions only in case config opt
    'suppress_errors_in_cleanup' is True.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if isinstance(exc_value, (lib_exc.NotFound, lib_exc.Forbidden)):
            return True
        elif CONF.suppress_errors_in_cleanup:
            LOG.error('Suppressed cleanup error: \n%s', traceback.format_exc())
            return True
        return False