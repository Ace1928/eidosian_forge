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
def _xml_to_container(self, node):
    """
        Converts a container XML node to a container instance

        :param node: XML info of the container
        :type node: :class:`xml.etree.ElementTree.Element`

        :return: A container instance
        :rtype: :class:`Container`
        """
    name = node.findtext(fixxpath(xpath='Name'))
    props = node.find(fixxpath(xpath='Properties'))
    metadata = node.find(fixxpath(xpath='Metadata'))
    extra = {'url': node.findtext(fixxpath(xpath='Url')), 'last_modified': node.findtext(fixxpath(xpath='Last-Modified')), 'etag': props.findtext(fixxpath(xpath='Etag')), 'lease': {'status': props.findtext(fixxpath(xpath='LeaseStatus')), 'state': props.findtext(fixxpath(xpath='LeaseState')), 'duration': props.findtext(fixxpath(xpath='LeaseDuration'))}, 'meta_data': {}}
    if extra['etag']:
        extra['etag'] = extra['etag'].replace('"', '')
    if metadata is not None:
        for meta in list(metadata):
            extra['meta_data'][meta.tag] = meta.text
    return Container(name=name, extra=extra, driver=self)