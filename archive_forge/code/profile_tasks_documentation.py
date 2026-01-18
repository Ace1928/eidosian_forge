from __future__ import (absolute_import, division, print_function)
import collections
import time
from ansible.module_utils.six.moves import reduce
from ansible.plugins.callback import CallbackBase

        Logs the start of each task
        