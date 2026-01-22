from __future__ import (absolute_import, division, print_function)
import abc
import os
import platform
import re
import sys
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common._collections_compat import Mapping, Sequence
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE, BOOLEANS_FALSE
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.util import (  # noqa: F401, pylint: disable=unused-import
def _get_tls_config(fail_function, **kwargs):
    if 'ssl_version' in kwargs and LooseVersion(docker_version) >= LooseVersion('7.0.0b1'):
        ssl_version = kwargs.pop('ssl_version')
        if ssl_version is not None:
            fail_function('ssl_version is not compatible with Docker SDK for Python 7.0.0+. You are using Docker SDK for Python {docker_py_version}. The ssl_version option (value: {ssl_version}) has either been set directly or with the environment variable DOCKER_SSL_VERSION. Make sure it is not set, or switch to an older version of Docker SDK for Python.'.format(docker_py_version=docker_version, ssl_version=ssl_version))
    if 'assert_hostname' in kwargs and LooseVersion(docker_version) >= LooseVersion('7.0.0b1'):
        assert_hostname = kwargs.pop('assert_hostname')
        if assert_hostname is not None:
            fail_function('tls_hostname is not compatible with Docker SDK for Python 7.0.0+. You are using Docker SDK for Python {docker_py_version}. The tls_hostname option (value: {tls_hostname}) has either been set directly or with the environment variable DOCKER_TLS_HOSTNAME. Make sure it is not set, or switch to an older version of Docker SDK for Python.'.format(docker_py_version=docker_version, tls_hostname=assert_hostname))
    kwargs = dict(((k, v) for k, v in kwargs.items() if v is not None))
    try:
        tls_config = TLSConfig(**kwargs)
        return tls_config
    except TLSParameterError as exc:
        fail_function('TLS config error: %s' % exc)