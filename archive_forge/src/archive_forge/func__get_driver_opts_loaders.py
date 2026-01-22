import collections
import copy
import importlib.metadata
import json
import logging
import operator
import sys
import yaml
from oslo_config import cfg
from oslo_i18n import _message
import stevedore.named  # noqa
def _get_driver_opts_loaders(namespaces, driver_option_name):
    mgr = stevedore.named.NamedExtensionManager(namespace='oslo.config.opts.' + driver_option_name, names=namespaces, on_load_failure_callback=on_load_failure_callback, invoke_on_load=False)
    return [(e.name, e.plugin) for e in mgr]