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

Tests to validate correct validation of SSL server certificates.

Note that this test assumes two external dependencies are available:
  - A http proxy, which by default is assumed to be at host 'cache' and port
    3128.  This can be overridden with environment variables PROXY_HOST and
    PROXY_PORT, respectively.
  - An ssl-enabled web server that will return a valid certificate signed by one
    of the bundled CAs, and which can be reached by an alternate hostname that
    does not match the CN in that certificate.  By default, this test uses host
    'www' (without fully qualified domain). This can be overridden with
    environment variable INVALID_HOSTNAME_HOST. If no suitable host is already
    available, such a mapping can be established by temporarily adding an IP
    address for, say, www.google.com or www.amazon.com to /etc/hosts.
