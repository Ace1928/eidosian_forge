from __future__ import absolute_import, division, print_function
from os import environ
from urllib.parse import urljoin
import platform
def _env_deprecation_warning(module, old_env, new_env, vers):
    if old_env in environ:
        if new_env in environ:
            module.warn(f'{old_env} env variable is ignored because {new_env} is specified. {old_env} env variable is deprecated and will be removed in version {vers} Please use {new_env} env variable only.')
        else:
            module.warn(f'{old_env} env variable is deprecated and will be removed in version {vers} Please use {new_env} env variable instead.')