import base64
import binascii
import os
import re
from boto.compat import StringIO, six
from boto.exception import BotoClientError
from boto.s3.key import Key as S3Key
from boto.s3.keyfile import KeyFile
from boto.utils import compute_hash, get_utf8able_str
Create a new object from a sequence of existing objects.

        The content of the object representing this Key will be the
        concatenation of the given object sequence. For more detail, visit

            https://developers.google.com/storage/docs/composite-objects

        :type components list of Keys
        :param components List of gs.Keys representing the component objects

        :type content_type (optional) string
        :param content_type Content type for the new composite object.
        