import re
import copy
import time
import base64
import warnings
from typing import List
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, basestring, ensure_string
from libcloud.utils.xml import findall, findattr, findtext, fixxpath
from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, AWSBaseResponse, SignedAWSConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date, parse_date_allow_empty
from libcloud.utils.publickey import get_pubkey_comment, get_pubkey_ssh2_fingerprint
from libcloud.compute.providers import Provider
from libcloud.compute.constants.ec2_region_details_partial import (
def _get_disk_container_params(self, disk_container):
    """
        Return a list of dictionaries with query parameters for
        a valid disk container.

        :param      disk_container: List of dictionaries with
                                    disk_container details
        :type       disk_container: ``list`` or ``dict``

        :return:    Dictionary representation of the disk_container
        :rtype:     ``dict``
        """
    if not isinstance(disk_container, (list, tuple)):
        raise AttributeError('disk_container not list or tuple')
    params = {}
    for idx, content in enumerate(disk_container):
        idx += 1
        if not isinstance(content, dict):
            raise AttributeError('content %s in disk_container not a dict' % content)
        for k, v in content.items():
            if not isinstance(v, dict):
                params['DiskContainer.%s' % k] = str(v)
            else:
                for key, value in v.items():
                    params['DiskContainer.{}.{}'.format(k, key)] = str(value)
    return params