import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetRuleShellFlags(self, rule):
    """Return RuleShellFlags about how the given rule should be run. This
        includes whether it should run under cygwin (msvs_cygwin_shell), and
        whether the commands should be quoted (msvs_quote_cmd)."""
    cygwin = int(rule.get('msvs_cygwin_shell', self.spec.get('msvs_cygwin_shell', 1))) != 0
    quote_cmd = int(rule.get('msvs_quote_cmd', 1))
    assert quote_cmd != 0 or cygwin != 1, 'msvs_quote_cmd=0 only applicable for msvs_cygwin_shell=0'
    return MsvsSettings.RuleShellFlags(cygwin, quote_cmd)