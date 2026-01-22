import logging
import os
from oslo_config import cfg
from oslo_middleware import cors
from oslo_policy import opts
from oslo_policy import policy
from paste import deploy
from glance.i18n import _
from glance.version import version_info as version
def _get_deployment_flavor(flavor=None):
    """
    Retrieve the paste_deploy.flavor config item, formatted appropriately
    for appending to the application name.

    :param flavor: if specified, use this setting rather than the
                   paste_deploy.flavor configuration setting
    """
    if not flavor:
        flavor = CONF.paste_deploy.flavor
    return '' if not flavor else '-' + flavor