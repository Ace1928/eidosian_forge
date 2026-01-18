import argparse
import fcntl
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from oslo_config import cfg
from oslo_utils import units
from glance.common import config
from glance.i18n import _
def write_pid_file(pid_file, pid):
    with open(pid_file, 'w') as fp:
        fp.write('%d\n' % pid)