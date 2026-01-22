import os
import hmac
import base64
import hashlib
import binascii
from datetime import datetime, timedelta
from libcloud.utils.py3 import ET, b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import fixxpath
from libcloud.utils.files import read_in_chunks
from libcloud.common.azure import AzureConnection, AzureActiveDirectoryConnection
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
class AzureBlobLease:
    """
    A class to help in leasing an azure blob and renewing the lease
    """

    def __init__(self, driver, object_path, use_lease):
        """
        :param driver: The Azure storage driver that is being used
        :type driver: :class:`AzureStorageDriver`

        :param object_path: The path of the object we need to lease
        :type object_path: ``str``

        :param use_lease: Indicates if we must take a lease or not
        :type use_lease: ``bool``
        """
        self.object_path = object_path
        self.driver = driver
        self.use_lease = use_lease
        self.lease_id = None
        self.params = {'comp': 'lease'}

    def renew(self):
        """
        Renew the lease if it is older than a predefined time period
        """
        if self.lease_id is None:
            return
        headers = {'x-ms-lease-action': 'renew', 'x-ms-lease-id': self.lease_id, 'x-ms-lease-duration': '60'}
        response = self.driver.connection.request(self.object_path, headers=headers, params=self.params, method='PUT')
        if response.status != httplib.OK:
            raise LibcloudError('Unable to obtain lease', driver=self)

    def update_headers(self, headers):
        """
        Update the lease id in the headers
        """
        if self.lease_id:
            headers['x-ms-lease-id'] = self.lease_id

    def __enter__(self):
        if not self.use_lease:
            return self
        headers = {'x-ms-lease-action': 'acquire', 'x-ms-lease-duration': '60'}
        response = self.driver.connection.request(self.object_path, headers=headers, params=self.params, method='PUT')
        if response.status == httplib.NOT_FOUND:
            return self
        elif response.status != httplib.CREATED:
            raise LibcloudError('Unable to obtain lease', driver=self)
        self.lease_id = response.headers['x-ms-lease-id']
        return self

    def __exit__(self, type, value, traceback):
        if self.lease_id is None:
            return
        headers = {'x-ms-lease-action': 'release', 'x-ms-lease-id': self.lease_id}
        response = self.driver.connection.request(self.object_path, headers=headers, params=self.params, method='PUT')
        if response.status != httplib.OK:
            raise LibcloudError('Unable to release lease', driver=self)