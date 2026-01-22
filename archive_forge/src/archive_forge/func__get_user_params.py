from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
def _get_user_params(params):
    return (params.get(ParamName.DATA) or {}, params.get(ParamName.QUERY_PARAMS) or {}, params.get(ParamName.PATH_PARAMS) or {})