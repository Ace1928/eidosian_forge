import io
import logging
import math
import re
import urllib
import eventlet
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import units
import glance_store
from glance_store import capabilities
from glance_store.common import utils
import glance_store.driver
from glance_store import exceptions
from glance_store.i18n import _
import glance_store.location
def _create_s3_client(self, loc):
    """Create a client object to use when connecting to S3.

        :param loc: `glance_store.location.Location` object, supplied
                    from glance_store.location.get_location_from_uri()
        :returns: An object with credentials to connect to S3
        """
    s3_host = self._option_get('s3_store_host')
    url_format = self._option_get('s3_store_bucket_url_format')
    calling_format = {'addressing_style': url_format}
    session = boto_session.Session(aws_access_key_id=loc.accesskey, aws_secret_access_key=loc.secretkey)
    config = boto_client.Config(s3=calling_format)
    location = get_s3_location(s3_host)
    bucket_name = loc.bucket
    if url_format == 'virtual' and (not boto_utils.check_dns_name(bucket_name)):
        raise boto_exceptions.InvalidDNSNameError(bucket_name=bucket_name)
    region_name, endpoint_url = (None, None)
    if self.region_name:
        region_name = self.region_name
        endpoint_url = s3_host
    elif location:
        region_name = location
    else:
        endpoint_url = s3_host
    return session.client(service_name='s3', endpoint_url=endpoint_url, region_name=region_name, use_ssl=loc.scheme == 's3+https', config=config)