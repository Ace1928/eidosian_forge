import copy
import pickle
import os
from tests.compat import unittest, mock
from tests.unit import MockServiceWithConfigTestCase
from nose.tools import assert_equal
from boto.auth import HmacAuthV4Handler
from boto.auth import S3HmacAuthV4Handler
from boto.auth import detect_potential_s3sigv4
from boto.auth import detect_potential_sigv4
from boto.connection import HTTPRequest
from boto.provider import Provider
from boto.regioninfo import RegionInfo
def test_s3_special_domain_signature_version():
    special_domains = ['storage.googleapis.com', 'mycustomdomain.example.com', 's3.amazonaws.com.example.com', 'mycustomdomain.example.com/amazonaws.com']
    for domain in special_domains:
        yield S3SignatureVersionTestCase(domain, 'nope').run