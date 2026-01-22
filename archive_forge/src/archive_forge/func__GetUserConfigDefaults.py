from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import locale
import os
import re
import signal
import subprocess
from googlecloudsdk.core.util import encoding
import six
def _GetUserConfigDefaults(self):
    """Consults the user shell config for defaults."""
    self._SendCommand('COSHELL_VERSION={coshell_version};_status() {{ return $1; }};[[ -f $HOME/.bashrc ]] && source $HOME/.bashrc;trap \'echo $?{exit} >&{fdstatus}\' 0;trap ":" 2;{get_completions_init}'.format(coshell_version=COSHELL_VERSION, exit=self.SHELL_STATUS_EXIT, fdstatus=self.SHELL_STATUS_FD, get_completions_init=_GET_COMPLETIONS_INIT))
    self._SendCommand('set -o monitor 2>/dev/null')
    self._SendCommand('shopt -s expand_aliases 2>/dev/null')
    self._GetModes()
    self._SendCommand('true')