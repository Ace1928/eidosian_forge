import collections.abc
import json
import typing as ty
from urllib import parse
from urllib import request
from openstack import exceptions
from openstack.orchestration.util import environment_format
from openstack.orchestration.util import template_format
from openstack.orchestration.util import utils
Handles any resource URLs specified in an environment.

    :param resource_registry: mapping of type name to template filename
    :type  resource_registry: dict
    :param files: dict to store loaded file contents into
    :type  files: dict
    :param env_base_url: base URL to look in when loading files
    :type  env_base_url: str or None
    