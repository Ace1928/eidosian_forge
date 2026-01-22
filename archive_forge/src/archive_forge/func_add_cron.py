import boto.pyami.installers
import os
import os.path
import stat
import boto
import random
from pwd import getpwnam
def add_cron(self, name, command, minute='*', hour='*', mday='*', month='*', wday='*', who='root', env=None):
    """
        Write a file to /etc/cron.d to schedule a command
            env is a dict containing environment variables you want to set in the file
            name will be used as the name of the file
        """
    if minute == 'random':
        minute = str(random.randrange(60))
    if hour == 'random':
        hour = str(random.randrange(24))
    fp = open('/etc/cron.d/%s' % name, 'w')
    if env:
        for key, value in env.items():
            fp.write('%s=%s\n' % (key, value))
    fp.write('%s %s %s %s %s %s %s\n' % (minute, hour, mday, month, wday, who, command))
    fp.close()