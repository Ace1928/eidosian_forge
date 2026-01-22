from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
class CompleteUserAtHost(Completer):
    """
    A completion action which produces matches in any of these forms::
        <username>
        <hostname>
        <username>@<hostname>
    """
    _descr = 'host | user@host'

    def _shellCode(self, optName, shellType):
        if shellType == _ZSH:
            return '%s:%s:{_ssh;if compset -P "*@"; then _wanted hosts expl "remote host name" _ssh_hosts && ret=0 elif compset -S "@*"; then _wanted users expl "login name" _ssh_users -S "" && ret=0 else if (( $+opt_args[-l] )); then tmp=() else tmp=( "users:login name:_ssh_users -qS@" ) fi; _alternative "hosts:remote host name:_ssh_hosts" "$tmp[@]" && ret=0 fi}' % (self._repeatFlag, self._description(optName))
        raise NotImplementedError(f'Unknown shellType {shellType!r}')