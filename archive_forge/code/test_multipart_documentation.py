import os
import unittest
import time
from boto.compat import StringIO
import mock
import boto
from boto.s3.connection import S3Connection

Some unit tests for the S3 MultiPartUpload
