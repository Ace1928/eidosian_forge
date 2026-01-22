import sys
from oslo_config import cfg
from oslo_policy import opts
from oslo_policy import policy
Verify context is service role according to global policy settings.

    :param context: The context object.
    :returns: True if the context is service role (as per the global
    enforcer) and False otherwise.
    