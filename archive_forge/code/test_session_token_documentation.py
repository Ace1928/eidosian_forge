import unittest
import os
from boto.exception import BotoServerError
from boto.sts.connection import STSConnection
from boto.sts.credentials import Credentials
from boto.s3.connection import S3Connection

Tests for Session Tokens
