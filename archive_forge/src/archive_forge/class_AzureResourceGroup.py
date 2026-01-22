import os
import time
import base64
import binascii
from libcloud.utils import iso8601
from libcloud.utils.py3 import parse_qs, urlparse, basestring
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.storage.types import ObjectDoesNotExistError
from libcloud.common.azure_arm import AzureResourceManagementConnection
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import Provider
from libcloud.storage.drivers.azure_blobs import AzureBlobsStorageDriver
class AzureResourceGroup:
    """Represent an Azure resource group."""

    def __init__(self, id, name, location, extra):
        self.id = id
        self.name = name
        self.location = location
        self.extra = extra

    def __repr__(self):
        return '<AzureResourceGroup: id=%s, name=%s, location=%s ...>' % (self.id, self.name, self.location)