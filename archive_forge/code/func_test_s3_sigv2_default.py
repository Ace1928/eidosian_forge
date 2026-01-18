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
def test_s3_sigv2_default():
    sigv2_regions = ['ap-northeast-1', 'ap-southeast-1', 'ap-southeast-2', 'eu-west-1', 'external-1', 'sa-east-1', 'us-east-1', 'us-gov-west-1', 'us-west-1', 'us-west-2']
    for region in sigv2_regions:
        _yield_all_region_tests(region, expected_signature_version='nope')