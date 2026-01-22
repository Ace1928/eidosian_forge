import fnmatch
import logging
from oslo_config import cfg
from stevedore import dispatch
import yaml
from oslo_messaging.notify import notifier
def _get_notifier_config_file(self, filename):
    """Broken out for testing."""
    return open(filename, 'r')