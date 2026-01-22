import os
import getpass
import json
import pty
import random
import re
import select
import string
import subprocess
import time
from functools import wraps
from ansible_collections.amazon.aws.plugins.module_utils.botocore import HAS_BOTO3
from ansible.errors import AnsibleConnectionFailure
from ansible.errors import AnsibleError
from ansible.errors import AnsibleFileNotFound
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six.moves import xrange
from ansible.module_utils._text import to_bytes
from ansible.module_utils._text import to_text
from ansible.plugins.connection import ConnectionBase
from ansible.plugins.shell.powershell import _common_args
from ansible.utils.display import Display
def _get_bucket_endpoint(self):
    """
        Fetches the correct S3 endpoint and region for use with our bucket.
        If we don't explicitly set the endpoint then some commands will use the global
        endpoint and fail
        (new AWS regions and new buckets in a region other than the one we're running in)
        """
    region_name = self.get_option('region') or 'us-east-1'
    profile_name = self.get_option('profile') or ''
    self._vvvv('_get_bucket_endpoint: S3 (global)')
    tmp_s3_client = self._get_boto_client('s3', region_name=region_name, profile_name=profile_name)
    bucket_location = tmp_s3_client.get_bucket_location(Bucket=self.get_option('bucket_name'))
    bucket_region = bucket_location['LocationConstraint']
    if self.get_option('bucket_endpoint_url'):
        return (self.get_option('bucket_endpoint_url'), bucket_region)
    self._vvvv(f'_get_bucket_endpoint: S3 (bucket region) - {bucket_region}')
    s3_bucket_client = self._get_boto_client('s3', region_name=bucket_region, profile_name=profile_name)
    return (s3_bucket_client.meta.endpoint_url, s3_bucket_client.meta.region_name)