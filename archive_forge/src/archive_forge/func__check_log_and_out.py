import sys
from io import StringIO
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.repn.tests.lp_diff import lp_diff
def _check_log_and_out(LOG, OUT, offset, msg=None):
    sys.stdout.flush()
    sys.stderr.flush()
    msg = str(msg) + ': ' if msg else ''
    if LOG.getvalue():
        raise RuntimeError('FAIL: %sMessage logged to the Logger:\n>>>\n%s<<<' % (msg, LOG.getvalue()))
    if OUT.getvalue():
        raise RuntimeError('FAIL: %sMessage sent to stdout/stderr:\n>>>\n%s<<<' % (msg, OUT.getvalue()))