import os
import re
import shutil
import sys
class IpFilter(CommandFilter):
    """Specific filter for the ip utility to that does not match exec."""

    def match(self, userargs):
        if userargs[0] == 'ip':
            for a, b in zip(userargs[1:], userargs[2:]):
                if a in NETNS_VARS:
                    return b not in EXEC_VARS
            else:
                return True