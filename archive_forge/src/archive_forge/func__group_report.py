import sys
import threading
from IPython import get_ipython
from IPython.core.ultratb import AutoFormattedTB
from logging import error, debug
def _group_report(self, group, name):
    """Report summary for a given job group.

        Return True if the group had any elements."""
    if group:
        print('%s jobs:' % name)
        for job in group:
            print('%s : %s' % (job.num, job))
        print()
        return True