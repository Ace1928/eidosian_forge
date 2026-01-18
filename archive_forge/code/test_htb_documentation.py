from twisted.protocols import htb
from twisted.trial import unittest
from .test_pcp import DummyConsumer

        L{htb.Bucket.drip} returns C{True} if the bucket is empty after that drip.
        