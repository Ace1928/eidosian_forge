from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import pkgutil
import tempfile
import textwrap
import six
import boto
from boto import config
import boto.auth
from boto.exception import NoAuthHandlerFound
from boto.gs.connection import GSConnection
from boto.provider import Provider
from boto.pyami.config import BotoConfigLocations
import gslib
from gslib import context_config
from gslib.exception import CommandException
from gslib.utils import system_util
from gslib.utils.constants import DEFAULT_GCS_JSON_API_VERSION
from gslib.utils.constants import DEFAULT_GSUTIL_STATE_DIR
from gslib.utils.constants import SSL_TIMEOUT_SEC
from gslib.utils.constants import UTF8
from gslib.utils.unit_util import HumanReadableToBytes
from gslib.utils.unit_util import ONE_MIB
import httplib2
from oauth2client.client import HAS_CRYPTO
def ConfigureCertsFile():
    """Configures and returns the CA Certificates file.

  If one is already configured, use it. Otherwise, use the cert roots
  distributed with gsutil.

  Returns:
    string filename of the certs file to use.
  """
    certs_file = boto.config.get('Boto', 'ca_certificates_file', None)
    if certs_file == 'system':
        return None
    if not certs_file:
        global configured_certs_file, temp_certs_file
        if not configured_certs_file:
            configured_certs_file = os.path.abspath(os.path.join(gslib.GSLIB_DIR, 'data', 'cacerts.txt'))
            if not os.path.exists(configured_certs_file):
                certs_data = pkgutil.get_data('gslib', 'data/cacerts.txt')
                if not certs_data:
                    raise CommandException('Certificates file not found. Please reinstall gsutil from scratch')
                certs_data = six.ensure_str(certs_data)
                fd, fname = tempfile.mkstemp(suffix='.txt', prefix='gsutil-cacerts')
                f = os.fdopen(fd, 'w')
                f.write(certs_data)
                f.close()
                temp_certs_file = fname
                configured_certs_file = temp_certs_file
        certs_file = configured_certs_file
    return certs_file