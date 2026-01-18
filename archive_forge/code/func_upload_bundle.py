import boto.ec2
from boto.mashups.iobject import IObject
from boto.pyami.config import BotoConfigPath, Config
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty, IntegerProperty, BooleanProperty, CalculatedProperty
from boto.manage import propget
from boto.ec2.zone import Zone
from boto.ec2.keypair import KeyPair
import os, time
from contextlib import closing
from boto.exception import EC2ResponseError
from boto.compat import six, StringIO
def upload_bundle(self, bucket, prefix, ssh_key):
    command = ''
    if self.uname != 'root':
        command = 'sudo '
    command += 'ec2-upload-bundle '
    command += '-m /mnt/%s.manifest.xml ' % prefix
    command += '-b %s ' % bucket
    command += '-a %s ' % self.server.ec2.aws_access_key_id
    command += '-s %s ' % self.server.ec2.aws_secret_access_key
    return command