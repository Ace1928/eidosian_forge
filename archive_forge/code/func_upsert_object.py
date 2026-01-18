from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
def upsert_object(self, op_name, params):
    """
        Updates an object if it already exists, or tries to create a new one if there is no
        such object. If multiple objects match filter criteria, or add operation is not supported,
        the exception is raised.

        :param op_name: upsert operation name
        :type op_name: str
        :param params: params that upsert operation should be executed with
        :type params: dict
        :return: upserted object representation
        :rtype: dict
        """

    def extract_and_validate_model():
        model = op_name[len(OperationNamePrefix.UPSERT):]
        if not self._conn.get_model_spec(model):
            raise FtdInvalidOperationNameError(op_name)
        return model
    model_name = extract_and_validate_model()
    model_operations = self.get_operation_specs_by_model_name(model_name)
    if not self._operation_checker.is_upsert_operation_supported(model_operations):
        raise FtdInvalidOperationNameError(op_name)
    existing_obj = self._find_object_matching_params(model_name, params)
    if existing_obj:
        equal_to_existing_obj = equal_objects(existing_obj, params[ParamName.DATA])
        return existing_obj if equal_to_existing_obj else self._edit_upserted_object(model_operations, existing_obj, params)
    else:
        return self._add_upserted_object(model_operations, params)