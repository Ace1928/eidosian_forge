import fnmatch
import logging
from oslo_config import cfg
from stevedore import dispatch
import yaml
from oslo_messaging.notify import notifier
def _should_load_plugin(self, ext, *args, **kwargs):
    if ext.name == 'routing':
        return False
    return ext.name in self.used_drivers