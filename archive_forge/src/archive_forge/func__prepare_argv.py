import errno
import os
import shlex
import signal
import sys
from collections import OrderedDict, UserList, defaultdict
from functools import partial
from subprocess import Popen
from time import sleep
from kombu.utils.encoding import from_utf8
from kombu.utils.objects import cached_property
from celery.platforms import IS_WINDOWS, Pidfile, signal_name
from celery.utils.nodenames import gethostname, host_format, node_format, nodesplit
from celery.utils.saferepr import saferepr
def _prepare_argv(self):
    cmd = self.expander(self.cmd).split(' ')
    i = cmd.index('celery') + 1
    options = self.options.copy()
    for opt, value in self.options.items():
        if opt in ('-A', '--app', '-b', '--broker', '--result-backend', '--loader', '--config', '--workdir', '-C', '--no-color', '-q', '--quiet'):
            cmd.insert(i, format_opt(opt, self.expander(value)))
            options.pop(opt)
    cmd = [' '.join(cmd)]
    argv = tuple(cmd + [format_opt(opt, self.expander(value)) for opt, value in options.items()] + [self.extra_args])
    if self.append:
        argv += (self.expander(self.append),)
    return argv