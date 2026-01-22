import os
import re
import shutil
import sys
class IpNetnsExecFilter(ChainingFilter):
    """Specific filter for the ip utility to that does match exec."""

    def match(self, userargs):
        if self.run_as != 'root' or len(userargs) < 4:
            return False
        return userargs[0] == 'ip' and userargs[1] in NETNS_VARS and (userargs[2] in EXEC_VARS)

    def exec_args(self, userargs):
        args = userargs[4:]
        if args:
            args[0] = os.path.basename(args[0])
        return args