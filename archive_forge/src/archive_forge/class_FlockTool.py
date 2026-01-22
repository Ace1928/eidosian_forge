import fcntl
import os
import struct
import subprocess
import sys
class FlockTool:
    """This class emulates the 'flock' command."""

    def Dispatch(self, args):
        """Dispatches a string command to a method."""
        if len(args) < 1:
            raise Exception('Not enough arguments')
        method = 'Exec%s' % self._CommandifyName(args[0])
        getattr(self, method)(*args[1:])

    def _CommandifyName(self, name_string):
        """Transforms a tool name like copy-info-plist to CopyInfoPlist"""
        return name_string.title().replace('-', '')

    def ExecFlock(self, lockfile, *cmd_list):
        """Emulates the most basic behavior of Linux's flock(1)."""
        fd = os.open(lockfile, os.O_WRONLY | os.O_NOCTTY | os.O_CREAT, 438)
        if sys.platform.startswith('aix'):
            op = struct.pack('hhIllqq', fcntl.F_WRLCK, 0, 0, 0, 0, 0, 0)
        else:
            op = struct.pack('hhllhhl', fcntl.F_WRLCK, 0, 0, 0, 0, 0, 0)
        fcntl.fcntl(fd, fcntl.F_SETLK, op)
        return subprocess.call(cmd_list)