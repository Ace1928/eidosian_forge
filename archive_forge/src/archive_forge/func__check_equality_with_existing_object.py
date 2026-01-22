from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
def _check_equality_with_existing_object(self, operation_name, params, e):
    """
        Looks for an existing object that caused "object duplicate" error and
        checks whether it corresponds to the one specified in `params`.

        In case a single object is found and it is equal to one we are trying
        to create, the existing object is returned.

        When the existing object is not equal to the object being created or
        several objects are returned, an exception is raised.
        """
    model_name = self.get_operation_spec(operation_name)[OperationField.MODEL_NAME]
    existing_obj = self._find_object_matching_params(model_name, params)
    if existing_obj is not None:
        if equal_objects(existing_obj, params[ParamName.DATA]):
            return existing_obj
        else:
            raise FtdConfigurationError(DUPLICATE_ERROR, existing_obj)
    raise e