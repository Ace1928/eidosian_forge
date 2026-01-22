from __future__ import (absolute_import, division, print_function)
import bisect
import json
import pkgutil
import re
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.distro import LinuxDistribution
from ansible.utils.display import Display
from ansible.utils.plugin_docs import get_versioned_doclink
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.facts.system.distribution import Distribution
from traceback import format_exc
class InterpreterDiscoveryRequiredError(Exception):

    def __init__(self, message, interpreter_name, discovery_mode):
        super(InterpreterDiscoveryRequiredError, self).__init__(message)
        self.interpreter_name = interpreter_name
        self.discovery_mode = discovery_mode

    def __str__(self):
        return self.message

    def __repr__(self):
        return self.message