from __future__ import (absolute_import, division, print_function)
import sys
import glob
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.utils import get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector

        Example contents /sys/class/fc_host/*/port_name:

        0x21000014ff52a9bb

        