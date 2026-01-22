from twisted.protocols import htb
from twisted.trial import unittest
from .test_pcp import DummyConsumer
class SomeBucket(htb.Bucket):
    maxburst = 100
    rate = 2