import sys
import string
import unittest
from unittest.mock import Mock, patch
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.nfsn import NFSNConnection
Check that X-NFSN-Authentication is set