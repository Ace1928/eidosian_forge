from __future__ import (absolute_import, division, print_function)
from ansible.plugins.callback import CallbackBase
from ansible import constants as C

    This is the default callback interface, which simply prints messages
    to stdout when new callback events are received.
    