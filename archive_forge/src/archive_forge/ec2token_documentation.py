import hashlib
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
import requests
import webob
from heat.api.aws import exception
from heat.common import endpoint_utils
from heat.common.i18n import _
from heat.common import wsgi
Extract the access key identifier.

        For v 0/1/2/3 this is passed as the AccessKeyId parameter,
        for version4 it is either and X-Amz-Credential parameter or a
        Credential= field in the 'Authorization' header string.
        