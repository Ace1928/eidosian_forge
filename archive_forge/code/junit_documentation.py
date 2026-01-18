from __future__ import (absolute_import, division, print_function)
import os
import time
import re
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.plugins.callback import CallbackBase
from ansible.utils._junit_xml import (
 generate a TestSuite report from the collected TaskData and HostData 