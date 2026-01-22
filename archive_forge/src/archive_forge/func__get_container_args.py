from unittest import mock
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.tests.unit.v1 import shell_test_base
from zunclient.v1 import containers_shell
def _get_container_args(**kwargs):
    default_args = {'auto_remove': False, 'environment': {}, 'hints': {}, 'labels': {}, 'mounts': [], 'nets': [], 'command': []}
    default_args.update(kwargs)
    return default_args