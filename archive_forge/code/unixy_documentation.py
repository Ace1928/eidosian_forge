from __future__ import (absolute_import, division, print_function)
from os.path import basename
from ansible import constants as C
from ansible import context
from ansible.module_utils.common.text.converters import to_text
from ansible.utils.color import colorize, hostcolor
from ansible.plugins.callback.default import CallbackModule as CallbackModule_default

    Design goals:
    - Print consolidated output that looks like a *NIX startup log
    - Defaults should avoid displaying unnecessary information wherever possible

    TODOs:
    - Only display task names if the task runs on at least one host
    - Add option to display all hostnames on a single line in the appropriate result color (failures may have a separate line)
    - Consolidate stats display
    - Don't show play name if no hosts found
    