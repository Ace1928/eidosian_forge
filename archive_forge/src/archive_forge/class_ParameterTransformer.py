import copy
from boto3.compat import collections_abc
from boto3.dynamodb.types import TypeSerializer, TypeDeserializer
from boto3.dynamodb.conditions import ConditionBase
from boto3.dynamodb.conditions import ConditionExpressionBuilder
from boto3.docs.utils import DocumentModifiedShape
class ParameterTransformer(object):
    """Transforms the input to and output from botocore based on shape"""

    def transform(self, params, model, transformation, target_shape):
        """Transforms the dynamodb input to or output from botocore

        It applies a specified transformation whenever a specific shape name
        is encountered while traversing the parameters in the dictionary.

        :param params: The parameters structure to transform.
        :param model: The operation model.
        :param transformation: The function to apply the parameter
        :param target_shape: The name of the shape to apply the
            transformation to
        """
        self._transform_parameters(model, params, transformation, target_shape)

    def _transform_parameters(self, model, params, transformation, target_shape):
        type_name = model.type_name
        if type_name in ['structure', 'map', 'list']:
            getattr(self, '_transform_%s' % type_name)(model, params, transformation, target_shape)

    def _transform_structure(self, model, params, transformation, target_shape):
        if not isinstance(params, collections_abc.Mapping):
            return
        for param in params:
            if param in model.members:
                member_model = model.members[param]
                member_shape = member_model.name
                if member_shape == target_shape:
                    params[param] = transformation(params[param])
                else:
                    self._transform_parameters(member_model, params[param], transformation, target_shape)

    def _transform_map(self, model, params, transformation, target_shape):
        if not isinstance(params, collections_abc.Mapping):
            return
        value_model = model.value
        value_shape = value_model.name
        for key, value in params.items():
            if value_shape == target_shape:
                params[key] = transformation(value)
            else:
                self._transform_parameters(value_model, params[key], transformation, target_shape)

    def _transform_list(self, model, params, transformation, target_shape):
        if not isinstance(params, collections_abc.MutableSequence):
            return
        member_model = model.member
        member_shape = member_model.name
        for i, item in enumerate(params):
            if member_shape == target_shape:
                params[i] = transformation(item)
            else:
                self._transform_parameters(member_model, params[i], transformation, target_shape)