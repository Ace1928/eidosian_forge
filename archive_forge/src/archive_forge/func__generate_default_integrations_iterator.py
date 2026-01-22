from __future__ import absolute_import
from threading import Lock
from sentry_sdk._compat import iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import logger
def _generate_default_integrations_iterator(integrations, auto_enabling_integrations):

    def iter_default_integrations(with_auto_enabling_integrations):
        """Returns an iterator of the default integration classes:"""
        from importlib import import_module
        if with_auto_enabling_integrations:
            all_import_strings = integrations + auto_enabling_integrations
        else:
            all_import_strings = integrations
        for import_string in all_import_strings:
            try:
                module, cls = import_string.rsplit('.', 1)
                yield getattr(import_module(module), cls)
            except (DidNotEnable, SyntaxError) as e:
                logger.debug('Did not import default integration %s: %s', import_string, e)
    if isinstance(iter_default_integrations.__doc__, str):
        for import_string in integrations:
            iter_default_integrations.__doc__ += '\n- `{}`'.format(import_string)
    return iter_default_integrations