import boto
from boto.manage.volume import Volume
from boto.exception import EC2ResponseError
import os, time
from boto.pyami.installers.ubuntu.installer import Installer
from string import Template
import boto
from boto.pyami.scriptbase import ScriptBase
import traceback
import boto
from boto.manage.volume import Volume
import boto
def update_fstab(self):
    f = open('/etc/fstab', 'a')
    f.write('%s\t%s\txfs\tdefaults 0 0\n' % (self.device, self.mount_point))
    f.close()