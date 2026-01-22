import os
import pwd
import re
import subprocess
import time
import xml.dom.minidom
import random
from ... import logging
from ...interfaces.base import CommandLine
from .base import SGELikeBatchManagerBase, logger
@staticmethod
def _qacct_verified_complete(taskid):
    """request definitive job completion information for the current job
        from the qacct report
        """
    sge_debug_print('WARNING:  CONTACTING qacct for finished jobs, {0}: {1}'.format(time.time(), 'Verifying Completion'))
    this_command = 'qacct'
    qacct_retries = 10
    is_complete = False
    while qacct_retries > 0:
        qacct_retries -= 1
        try:
            proc = subprocess.Popen([this_command, '-o', pwd.getpwuid(os.getuid())[0], '-j', str(taskid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            qacct_result, _ = proc.communicate()
            if qacct_result.find(str(taskid)):
                is_complete = True
            sge_debug_print('NOTE: qacct for jobs\n{0}'.format(qacct_result))
            break
        except:
            sge_debug_print('NOTE: qacct call failed')
            time.sleep(5)
            pass
    return is_complete