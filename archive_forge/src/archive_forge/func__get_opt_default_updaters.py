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
def _get_opt_default_updaters(namespaces):
    mgr = stevedore.named.NamedExtensionManager('oslo.config.opts.defaults', names=namespaces, warn_on_missing_entrypoint=False, on_load_failure_callback=on_load_failure_callback, invoke_on_load=False)
    return [ep.plugin for ep in mgr]