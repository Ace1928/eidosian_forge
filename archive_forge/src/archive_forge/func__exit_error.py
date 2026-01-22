import configparser
import logging
import os
import sys
from oslo_rootwrap import subprocess
from oslo_rootwrap import wrapper
def _exit_error(execname, message, errorcode, log=True):
    print('%s: %s' % (execname, message), file=sys.stderr)
    if log:
        logging.error(message)
    sys.exit(errorcode)