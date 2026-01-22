import datetime
import logging
import logging.handlers
import os
import re
import socket
import sys
import threading
import ovs.dirs
import ovs.unixctl
import ovs.util
@staticmethod
def _unixctl_vlog_close(conn, unused_argv, unused_aux):
    if Vlog.__log_file:
        if sys.platform != 'win32':
            logger = logging.getLogger('file')
            logger.removeHandler(Vlog.__file_handler)
        else:
            Vlog.close_log_file()
    conn.reply(None)