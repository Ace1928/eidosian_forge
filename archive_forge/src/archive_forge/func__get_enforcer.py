import logging
import sys
import textwrap
import warnings
import yaml
from oslo_config import cfg
from oslo_serialization import jsonutils
import stevedore
from oslo_policy import policy
def _get_enforcer(namespace):
    """Find a policy.Enforcer via an entry point with the given namespace.

    :param namespace: a namespace under oslo.policy.enforcer where the desired
                      enforcer object can be found.
    :returns: a policy.Enforcer object
    """
    mgr = stevedore.named.NamedExtensionManager('oslo.policy.enforcer', names=[namespace], on_load_failure_callback=on_load_failure_callback, invoke_on_load=True)
    if namespace not in mgr:
        raise KeyError('Namespace "%s" not found.' % namespace)
    enforcer = mgr[namespace].obj
    return enforcer