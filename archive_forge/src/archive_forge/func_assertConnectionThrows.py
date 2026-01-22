import os
import ssl
import unittest
import mock
from nose.plugins.attrib import attr
import boto
from boto.pyami.config import Config
from boto import exception, https_connection
from boto.gs.connection import GSConnection
from boto.s3.connection import S3Connection
def assertConnectionThrows(self, connection_class, error):
    conn = connection_class('fake_id', 'fake_secret')
    self.assertRaises(error, conn.get_all_buckets)