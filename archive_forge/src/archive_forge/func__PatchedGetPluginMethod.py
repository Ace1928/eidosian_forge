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
def _PatchedGetPluginMethod(cls, requested_capability=None):
    handler_subclasses = orig_get_plugin_method(cls, requested_capability=requested_capability)
    xml_oauth2_handlers = (gcs_oauth2_boto_plugin.oauth2_plugin.OAuth2ServiceAccountAuth, gcs_oauth2_boto_plugin.oauth2_plugin.OAuth2Auth)
    new_result = sorted([r for r in handler_subclasses if r not in xml_oauth2_handlers], key=lambda handler_t: handler_t.__name__) + sorted([r for r in handler_subclasses if r in xml_oauth2_handlers], key=lambda handler_t: handler_t.__name__)
    return new_result