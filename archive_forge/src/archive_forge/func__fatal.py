import errno
import os
import signal
import sys
import time
import ovs.dirs
import ovs.fatal_signal
import ovs.process
import ovs.socket_util
import ovs.timeval
import ovs.util
import ovs.vlog
def _fatal(msg):
    vlog.err(msg)
    sys.stderr.write('%s\n' % msg)
    sys.exit(1)