import base64
import copy
import logging
import os
import re
import uuid
import warnings
from io import BytesIO
import botocore
import botocore.auth
from botocore import utils
from botocore.compat import (
from botocore.docs.utils import (
from botocore.endpoint_provider import VALID_HOST_LABEL_RE
from botocore.exceptions import (
from botocore.regions import EndpointResolverBuiltins
from botocore.signers import (
from botocore.utils import (
from botocore import retryhandler  # noqa
from botocore import translate  # noqa
from botocore.compat import MD5_AVAILABLE  # noqa
from botocore.exceptions import MissingServiceIdError  # noqa
from botocore.utils import hyphenize_service_id  # noqa
from botocore.utils import is_global_accesspoint  # noqa
from botocore.utils import SERVICE_NAME_ALIASES  # noqa
def _decode_policy_types(parsed, shape):
    shape_name = 'policyDocumentType'
    if shape.type_name == 'structure':
        for member_name, member_shape in shape.members.items():
            if member_shape.type_name == 'string' and member_shape.name == shape_name and (member_name in parsed):
                parsed[member_name] = decode_quoted_jsondoc(parsed[member_name])
            elif member_name in parsed:
                _decode_policy_types(parsed[member_name], member_shape)
    if shape.type_name == 'list':
        shape_member = shape.member
        for item in parsed:
            _decode_policy_types(item, shape_member)