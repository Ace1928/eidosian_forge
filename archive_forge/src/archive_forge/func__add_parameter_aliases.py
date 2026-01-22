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
def _add_parameter_aliases(handler_list):
    aliases = {'ec2.*.Filter': 'Filters', 'logs.CreateExportTask.from': 'fromTime', 'cloudsearchdomain.Search.return': 'returnFields'}
    for original, new_name in aliases.items():
        event_portion, original_name = original.rsplit('.', 1)
        parameter_alias = ParameterAlias(original_name, new_name)
        parameter_build_event_handler_tuple = ('before-parameter-build.' + event_portion, parameter_alias.alias_parameter_in_call, REGISTER_FIRST)
        docs_event_handler_tuple = ('docs.*.' + event_portion + '.complete-section', parameter_alias.alias_parameter_in_documentation)
        handler_list.append(parameter_build_event_handler_tuple)
        handler_list.append(docs_event_handler_tuple)