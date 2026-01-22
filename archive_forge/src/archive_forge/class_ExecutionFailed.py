import sys
import subprocess
import py
from subprocess import Popen, PIPE
class ExecutionFailed(py.error.Error):

    def __init__(self, status, systemstatus, cmd, out, err):
        Exception.__init__(self)
        self.status = status
        self.systemstatus = systemstatus
        self.cmd = cmd
        self.err = err
        self.out = out

    def __str__(self):
        return 'ExecutionFailed: %d  %s\n%s' % (self.status, self.cmd, self.err)