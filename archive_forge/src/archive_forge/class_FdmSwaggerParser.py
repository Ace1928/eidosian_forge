from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
class FdmSwaggerParser:
    _definitions = None
    _base_path = None

    def parse_spec(self, spec, docs=None):
        """
        This method simplifies a swagger format, resolves a model name for each operation, and adds documentation for
        each operation and model if it is provided.

        :param spec: An API specification in the swagger format, see
            <https://github.com/OAI/OpenAPI-Specification/blob/master/versions/2.0.md>
        :type spec: dict
        :param spec: A documentation map containing descriptions for models, operations and operation parameters.
        :type docs: dict
        :rtype: dict
        :return:
        Ex.
            The models field contains model definition from swagger see
            <#https://github.com/OAI/OpenAPI-Specification/blob/master/versions/2.0.md#definitions>
            {
                'models':{
                    'model_name':{...},
                    ...
                },
                'operations':{
                    'operation_name':{
                        'method': 'get', #post, put, delete
                        'url': '/api/fdm/v2/object/networks', #url already contains a value from `basePath`
                        'modelName': 'NetworkObject', # it is a link to the model from 'models'
                                                      # None - for a delete operation or we don't have information
                                                      # '_File' - if an endpoint works with files
                        'returnMultipleItems': False, # shows if the operation returns a single item or an item list
                        'parameters': {
                            'path':{
                                'param_name':{
                                    'type': 'string'#integer, boolean, number
                                    'required' True #False
                                }
                                ...
                                },
                            'query':{
                                'param_name':{
                                    'type': 'string'#integer, boolean, number
                                    'required' True #False
                                }
                                ...
                            }
                        }
                    },
                    ...
                },
                'model_operations':{
                    'model_name':{ # a list of operations available for the current model
                        'operation_name':{
                            ... # the same as in the operations section
                        },
                        ...
                    },
                    ...
                }
            }
        """
        self._definitions = spec[SpecProp.DEFINITIONS]
        self._base_path = spec[PropName.BASE_PATH]
        operations = self._get_operations(spec)
        if docs:
            operations = self._enrich_operations_with_docs(operations, docs)
            self._definitions = self._enrich_definitions_with_docs(self._definitions, docs)
        return {SpecProp.MODELS: self._definitions, SpecProp.OPERATIONS: operations, SpecProp.MODEL_OPERATIONS: self._get_model_operations(operations)}

    @property
    def base_path(self):
        return self._base_path

    def _get_model_operations(self, operations):
        model_operations = {}
        for operations_name, params in iteritems(operations):
            model_name = params[OperationField.MODEL_NAME]
            model_operations.setdefault(model_name, {})[operations_name] = params
        return model_operations

    def _get_operations(self, spec):
        paths_dict = spec[PropName.PATHS]
        operations_dict = {}
        for url, operation_params in iteritems(paths_dict):
            for method, params in iteritems(operation_params):
                operation = {OperationField.METHOD: method, OperationField.URL: self._base_path + url, OperationField.MODEL_NAME: self._get_model_name(method, params), OperationField.RETURN_MULTIPLE_ITEMS: self._return_multiple_items(params), OperationField.TAGS: params.get(OperationField.TAGS, [])}
                if OperationField.PARAMETERS in params:
                    operation[OperationField.PARAMETERS] = self._get_rest_params(params[OperationField.PARAMETERS])
                operation_id = params[PropName.OPERATION_ID]
                operations_dict[operation_id] = operation
        return operations_dict

    def _enrich_operations_with_docs(self, operations, docs):

        def get_operation_docs(op):
            op_url = op[OperationField.URL][len(self._base_path):]
            return docs[PropName.PATHS].get(op_url, {}).get(op[OperationField.METHOD], {})
        for operation in operations.values():
            operation_docs = get_operation_docs(operation)
            operation[OperationField.DESCRIPTION] = operation_docs.get(PropName.DESCRIPTION, '')
            if OperationField.PARAMETERS in operation:
                param_descriptions = dict(((p[PropName.NAME], p[PropName.DESCRIPTION]) for p in operation_docs.get(OperationField.PARAMETERS, {})))
                for param_name, params_spec in operation[OperationField.PARAMETERS][OperationParams.PATH].items():
                    params_spec[OperationField.DESCRIPTION] = param_descriptions.get(param_name, '')
                for param_name, params_spec in operation[OperationField.PARAMETERS][OperationParams.QUERY].items():
                    params_spec[OperationField.DESCRIPTION] = param_descriptions.get(param_name, '')
        return operations

    def _enrich_definitions_with_docs(self, definitions, docs):
        for model_name, model_def in definitions.items():
            model_docs = docs[SpecProp.DEFINITIONS].get(model_name, {})
            model_def[PropName.DESCRIPTION] = model_docs.get(PropName.DESCRIPTION, '')
            for prop_name, prop_spec in model_def.get(PropName.PROPERTIES, {}).items():
                prop_spec[PropName.DESCRIPTION] = model_docs.get(PropName.PROPERTIES, {}).get(prop_name, '')
                prop_spec[PropName.REQUIRED] = prop_name in model_def.get(PropName.REQUIRED, [])
        return definitions

    def _get_model_name(self, method, params):
        if method == HTTPMethod.GET:
            return self._get_model_name_from_responses(params)
        elif method == HTTPMethod.POST or method == HTTPMethod.PUT:
            return self._get_model_name_for_post_put_requests(params)
        elif method == HTTPMethod.DELETE:
            return self._get_model_name_from_delete_operation(params)
        else:
            return None

    @staticmethod
    def _return_multiple_items(op_params):
        """
        Defines if the operation returns one item or a list of items.

        :param op_params: operation specification
        :return: True if the operation returns a list of items, otherwise False
        """
        try:
            schema = op_params[PropName.RESPONSES][SUCCESS_RESPONSE_CODE][PropName.SCHEMA]
            return PropName.ITEMS in schema[PropName.PROPERTIES]
        except KeyError:
            return False

    def _get_model_name_from_delete_operation(self, params):
        operation_id = params[PropName.OPERATION_ID]
        if operation_id.startswith(DELETE_PREFIX):
            model_name = operation_id[len(DELETE_PREFIX):]
            if model_name in self._definitions:
                return model_name
        return None

    def _get_model_name_for_post_put_requests(self, params):
        model_name = None
        if OperationField.PARAMETERS in params:
            body_param_dict = self._get_body_param_from_parameters(params[OperationField.PARAMETERS])
            if body_param_dict:
                schema_ref = body_param_dict[PropName.SCHEMA][PropName.REF]
                model_name = self._get_model_name_byschema_ref(schema_ref)
        if model_name is None:
            model_name = self._get_model_name_from_responses(params)
        return model_name

    @staticmethod
    def _get_body_param_from_parameters(params):
        return next((param for param in params if param['in'] == 'body'), None)

    def _get_model_name_from_responses(self, params):
        responses = params[PropName.RESPONSES]
        if SUCCESS_RESPONSE_CODE in responses:
            response = responses[SUCCESS_RESPONSE_CODE][PropName.SCHEMA]
            if PropName.REF in response:
                return self._get_model_name_byschema_ref(response[PropName.REF])
            elif PropName.PROPERTIES in response:
                ref = response[PropName.PROPERTIES][PropName.ITEMS][PropName.ITEMS][PropName.REF]
                return self._get_model_name_byschema_ref(ref)
            elif PropName.TYPE in response and response[PropName.TYPE] == PropType.FILE:
                return FILE_MODEL_NAME
        else:
            return None

    def _get_rest_params(self, params):
        path = {}
        query = {}
        operation_param = {OperationParams.PATH: path, OperationParams.QUERY: query}
        for param in params:
            in_param = param['in']
            if in_param == OperationParams.QUERY:
                query[param[PropName.NAME]] = self._simplify_param_def(param)
            elif in_param == OperationParams.PATH:
                path[param[PropName.NAME]] = self._simplify_param_def(param)
        return operation_param

    @staticmethod
    def _simplify_param_def(param):
        return {PropName.TYPE: param[PropName.TYPE], PropName.REQUIRED: param[PropName.REQUIRED]}

    def _get_model_name_byschema_ref(self, schema_ref):
        model_name = _get_model_name_from_url(schema_ref)
        model_def = self._definitions[model_name]
        if PropName.ALL_OF in model_def:
            return self._get_model_name_byschema_ref(model_def[PropName.ALL_OF][0][PropName.REF])
        else:
            return model_name