import sys
import unittest
from libcloud.test import LibcloudTestCase
from libcloud.common import aliyun
from libcloud.common.aliyun import AliyunRequestSignerAlgorithmV1_0
class AliyunCommonTestCase(LibcloudTestCase):

    def test_percent_encode(self):
        data = {'abc': 'abc', ' *~': '%20%2A~'}
        for key in data:
            self.assertEqual(data[key], aliyun._percent_encode(key))