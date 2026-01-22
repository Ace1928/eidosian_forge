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
def _annotate_with_default_opts(self, options):
    options['-n'] = self.name
    self._setdefaultopt(options, ['--pidfile', '-p'], '/var/run/celery/%n.pid')
    self._setdefaultopt(options, ['--logfile', '-f'], '/var/log/celery/%n%I.log')
    self._setdefaultopt(options, ['--executable'], sys.executable)
    return options