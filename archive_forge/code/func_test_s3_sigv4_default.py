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
def test_s3_sigv4_default():
    sigv4_regions = ['ap-northeast-2', 'ap-south-1', 'ca-central-1', 'eu-central-1', 'eu-west-2', 'us-east-2']
    for region in sigv4_regions:
        _yield_all_region_tests(region)
    cn_regions = ['cn-north-1']
    for region in cn_regions:
        _yield_all_region_tests(region, dns_suffix='amazon.com.cn')
    _yield_all_region_tests('mars-west-1')