import os
import sys
from boto.utils import ShellCommand, get_ts
import boto
import boto.utils
def umount(self, path):
    if os.path.ismount(path):
        self.run('umount %s' % path)