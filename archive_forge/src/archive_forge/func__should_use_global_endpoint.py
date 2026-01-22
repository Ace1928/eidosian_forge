import base64
import datetime
import json
import weakref
import botocore
import botocore.auth
from botocore.awsrequest import create_request_object, prepare_request_dict
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import ArnParser, datetime2timestamp
from botocore.utils import fix_s3_host  # noqa
def _should_use_global_endpoint(client):
    if client.meta.partition != 'aws':
        return False
    s3_config = client.meta.config.s3
    if s3_config:
        if s3_config.get('use_dualstack_endpoint', False):
            return False
        if s3_config.get('us_east_1_regional_endpoint') == 'regional' and client.meta.config.region_name == 'us-east-1':
            return False
        if s3_config.get('addressing_style') == 'virtual':
            return False
    return True