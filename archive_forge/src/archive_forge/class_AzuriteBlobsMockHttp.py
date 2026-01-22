import os
import sys
import json
import tempfile
from io import BytesIO
from libcloud.test import generate_random_data  # pylint: disable-msg=E0611
from libcloud.test import unittest
from libcloud.utils.py3 import b, httplib, parse_qs, urlparse, basestring
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_AZURE_BLOBS_PARAMS, STORAGE_AZURITE_BLOBS_PARAMS
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.azure_blobs import (
class AzuriteBlobsMockHttp(AzureBlobsMockHttp):
    fixtures = StorageFileFixtures('azurite_blobs')

    def _get_method_name(self, *args, **kwargs):
        method_name = super()._get_method_name(*args, **kwargs)
        if method_name.startswith('_account'):
            method_name = method_name[8:]
        return method_name