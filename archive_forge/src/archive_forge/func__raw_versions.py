import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
def _raw_versions(self):
    return [block._raw_version for block in self._blocks]