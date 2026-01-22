import copy
from keystoneauth1 import loading
from oslo_config import cfg
from keystonemiddleware.auth_token import _base
Return a list of oslo_config options available in auth_token middleware.

    The returned list includes the non-deprecated oslo_config options which may
    be registered at runtime by the project. The purpose of this is to allow
    tools like the Oslo sample config file generator to discover the options
    exposed to users by this middleware.

    Deprecated Options should not show up here so as to not be included in
    sample configuration.

    Each element of the list is a tuple. The first element is the name of the
    group under which the list of elements in the second element will be
    registered. A group name of None corresponds to the [DEFAULT] group in
    config files.

    This function is discoverable via the entry point
    'keystonemiddleware.auth_token' under the 'oslo.config.opts' namespace.

    :returns: a list of (group_name, opts) tuples
    