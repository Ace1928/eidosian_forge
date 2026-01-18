from __future__ import (absolute_import, division, print_function)
import collections
import time
from ansible.plugins.callback import CallbackBase
from ansible.module_utils.six.moves import reduce

        Logs the start of each task
        