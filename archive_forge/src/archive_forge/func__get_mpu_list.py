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
@staticmethod
def _get_mpu_list(pedict):
    """Convert an object type and struct for use in
        boto3.client('s3').complete_multipart_upload.

        :param pedict: dict which containing UploadPart.etag
        :returns: list with pedict converted properly
        """
    return {'Parts': [{'PartNumber': pnum, 'ETag': etag} for pnum, etag in pedict.items()]}