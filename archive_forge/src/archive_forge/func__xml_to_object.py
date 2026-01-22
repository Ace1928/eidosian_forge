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
def _xml_to_object(self, container, blob):
    """
        Converts a BLOB XML node to an object instance

        :param container: Instance of the container holding the blob
        :type: :class:`Container`

        :param blob: XML info of the blob
        :type blob: L{}

        :return: An object instance
        :rtype: :class:`Object`
        """
    name = blob.findtext(fixxpath(xpath='Name'))
    props = blob.find(fixxpath(xpath='Properties'))
    metadata = blob.find(fixxpath(xpath='Metadata'))
    etag = props.findtext(fixxpath(xpath='Etag'))
    size = int(props.findtext(fixxpath(xpath='Content-Length')))
    extra = {'content_type': props.findtext(fixxpath(xpath='Content-Type')), 'etag': etag, 'md5_hash': props.findtext(fixxpath(xpath='Content-MD5')), 'last_modified': props.findtext(fixxpath(xpath='Last-Modified')), 'url': blob.findtext(fixxpath(xpath='Url')), 'hash': props.findtext(fixxpath(xpath='Etag')), 'lease': {'status': props.findtext(fixxpath(xpath='LeaseStatus')), 'state': props.findtext(fixxpath(xpath='LeaseState')), 'duration': props.findtext(fixxpath(xpath='LeaseDuration'))}, 'content_encoding': props.findtext(fixxpath(xpath='Content-Encoding')), 'content_language': props.findtext(fixxpath(xpath='Content-Language')), 'blob_type': props.findtext(fixxpath(xpath='BlobType'))}
    if extra['md5_hash']:
        value = binascii.hexlify(base64.b64decode(b(extra['md5_hash'])))
        value = value.decode('ascii')
        extra['md5_hash'] = value
    meta_data = {}
    if metadata is not None:
        for meta in list(metadata):
            meta_data[meta.tag] = meta.text
    return Object(name=name, size=size, hash=etag, meta_data=meta_data, extra=extra, container=container, driver=self)