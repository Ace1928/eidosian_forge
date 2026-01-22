import abc
import copy
import hashlib
import os
import ssl
import time
import uuid
import jwt.utils
import oslo_cache
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
import requests.auth
import webob.dec
import webob.exc
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from keystoneauth1.loading import session as session_loading
from keystonemiddleware._common import config
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.exceptions import ConfigurationError
from keystonemiddleware.exceptions import KeystoneMiddlewareException
from keystonemiddleware.i18n import _
def _read_data_from_token(self, token_metadata, config_key, is_required=False, value_type=None):
    """Read value from token metadata.

        Read the necessary information from the token metadata with the
        config key.
        """
    if not value_type:
        value_type = str
    meta_key = self._get_config_option(config_key, is_required=is_required)
    if not meta_key:
        return None
    if meta_key.find('.') >= 0:
        meta_value = None
        for temp_key in meta_key.split('.'):
            if not temp_key:
                self._log.critical('Configuration error. config_key: %s , meta_key: %s ' % (config_key, meta_key))
                raise ConfigurationError(_('Failed to parse the necessary information for the field "%s".') % meta_key)
            if not meta_value:
                meta_value = token_metadata.get(temp_key)
            else:
                if not isinstance(meta_value, dict):
                    self._log.warning('Failed to parse the necessary information. The meta_value is not of type dict.config_key: %s , meta_key: %s, value: %s' % (config_key, meta_key, meta_value))
                    raise ForbiddenToken(_('Failed to parse the necessary information for the field "%s".') % meta_key)
                meta_value = meta_value.get(temp_key)
    else:
        meta_value = token_metadata.get(meta_key)
    if not meta_value:
        if is_required:
            self._log.warning('Failed to parse the necessary information. The meta value is required.config_key: %s , meta_key: %s, value: %s, need_type: %s' % (config_key, meta_key, meta_value, value_type))
            raise ForbiddenToken(_('Failed to parse the necessary information for the field "%s".') % meta_key)
        else:
            meta_value = None
    elif not isinstance(meta_value, value_type):
        self._log.warning('Failed to parse the necessary information. The meta value is of an incorrect type.config_key: %s , meta_key: %s, value: %s, need_type: %s' % (config_key, meta_key, meta_value, value_type))
        raise ForbiddenToken(_('Failed to parse the necessary information for the field "%s".') % meta_key)
    return meta_value