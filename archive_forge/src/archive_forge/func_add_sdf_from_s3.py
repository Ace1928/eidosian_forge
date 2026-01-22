import boto.exception
from boto.compat import json
import requests
import boto
def add_sdf_from_s3(self, key_obj):
    """
        Load an SDF from S3

        Using this method will result in documents added through
        :func:`add` and :func:`delete` being ignored.

        :type key_obj: :class:`boto.s3.key.Key`
        :param key_obj: An S3 key which contains an SDF
        """
    self._sdf = key_obj.get_contents_as_string()