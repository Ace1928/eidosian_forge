import os
import sys
import time
import random
import urllib
import datetime
import unittest
import threading
from unittest import mock
import requests
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.google import (
class GoogleAuthTypeTest(GoogleTestCase):

    def test_guess(self):
        self.assertEqual(GoogleAuthType.guess_type(GCE_PARAMS_IA[0]), GoogleAuthType.IA)
        with mock.patch.object(GoogleAuthType, '_is_gce', return_value=True):
            self.assertEqual(GoogleAuthType.guess_type(GCE_PARAMS[0]), GoogleAuthType.SA)
            self.assertEqual(GoogleAuthType.guess_type(GCS_S3_PARAMS_20[0]), GoogleAuthType.GCS_S3)
            self.assertEqual(GoogleAuthType.guess_type(GCS_S3_PARAMS_24[0]), GoogleAuthType.GCS_S3)
            self.assertEqual(GoogleAuthType.guess_type(GCS_S3_PARAMS_61[0]), GoogleAuthType.GCS_S3)
            self.assertEqual(GoogleAuthType.guess_type(GCE_PARAMS_GCE[0]), GoogleAuthType.GCE)

    def test_guess_gce_metadata_server_not_called_for_ia(self):
        with mock.patch.object(GoogleAuthType, '_is_gce', return_value=False):
            self.assertEqual(GoogleAuthType._is_gce.call_count, 0)
            self.assertEqual(GoogleAuthType.guess_type(GCE_PARAMS_IA_2[0]), GoogleAuthType.IA)
            self.assertEqual(GoogleAuthType._is_gce.call_count, 0)