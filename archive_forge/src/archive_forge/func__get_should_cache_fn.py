import re
import ssl
import urllib.parse
import dogpile.cache
from dogpile.cache import api
from dogpile.cache import proxy
from dogpile.cache import util
from oslo_log import log
from oslo_utils import importutils
from oslo_cache._i18n import _
from oslo_cache import _opts
from oslo_cache import exception
def _get_should_cache_fn(conf, group):
    """Build a function that returns a config group's caching status.

    For any given object that has caching capabilities, a boolean config option
    for that object's group should exist and default to ``True``. This
    function will use that value to tell the caching decorator if caching for
    that object is enabled. To properly use this with the decorator, pass this
    function the configuration group and assign the result to a variable.
    Pass the new variable to the caching decorator as the named argument
    ``should_cache_fn``.

    :param conf: config object, must have had :func:`configure` called on it.
    :type conf: oslo_config.cfg.ConfigOpts
    :param group: name of the configuration group to examine
    :type group: string
    :returns: function reference
    """

    def should_cache(value):
        if not conf.cache.enabled:
            return False
        conf_group = getattr(conf, group)
        return getattr(conf_group, 'caching', True)
    return should_cache