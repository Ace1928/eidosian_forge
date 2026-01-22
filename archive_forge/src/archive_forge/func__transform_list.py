import copy
from boto3.compat import collections_abc
from boto3.dynamodb.types import TypeSerializer, TypeDeserializer
from boto3.dynamodb.conditions import ConditionBase
from boto3.dynamodb.conditions import ConditionExpressionBuilder
from boto3.docs.utils import DocumentModifiedShape
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