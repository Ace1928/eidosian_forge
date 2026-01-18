import time
from threading import Timer
from tests.unit import unittest
import boto
from boto.compat import StringIO
from boto.sqs.bigmessage import BigMessage
from boto.exception import SQSError

Some unit tests for the SQSConnection
