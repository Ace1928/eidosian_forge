from __future__ import (absolute_import, division, print_function)
import typing as t
from ansible.galaxy.api import GalaxyAPI, GalaxyError
from ansible.module_utils.common.text.converters import to_text
from ansible.utils.display import Display
def _assert_that_offline_mode_is_not_requested(self):
    if self.is_offline_mode_requested:
        raise NotImplementedError("The calling code is not supposed to be invoked in 'offline' mode.")