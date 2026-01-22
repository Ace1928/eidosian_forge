import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
def _get_bugs_closed_generic(self, type_re):
    changes = ' '.join(self._changes)
    bugs = []
    for match in type_re.finditer(changes):
        closes_list = match.group(0)
        for bugmatch in re.finditer('\\d+', closes_list):
            bugs.append(int(bugmatch.group(0)))
    return bugs