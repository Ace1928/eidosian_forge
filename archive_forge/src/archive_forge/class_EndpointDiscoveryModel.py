import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
class EndpointDiscoveryModel:

    def __init__(self, service_model):
        self._service_model = service_model

    @CachedProperty
    def discovery_operation_name(self):
        discovery_operation = self._service_model.endpoint_discovery_operation
        return xform_name(discovery_operation.name)

    @CachedProperty
    def discovery_operation_keys(self):
        discovery_operation = self._service_model.endpoint_discovery_operation
        keys = []
        if discovery_operation.input_shape:
            keys = list(discovery_operation.input_shape.members.keys())
        return keys

    def discovery_required_for(self, operation_name):
        try:
            operation_model = self._service_model.operation_model(operation_name)
            return operation_model.endpoint_discovery.get('required', False)
        except OperationNotFoundError:
            return False

    def discovery_operation_kwargs(self, **kwargs):
        input_keys = self.discovery_operation_keys
        if not kwargs.get('Identifiers'):
            kwargs.pop('Operation', None)
            kwargs.pop('Identifiers', None)
        return {k: v for k, v in kwargs.items() if k in input_keys}

    def gather_identifiers(self, operation, params):
        return self._gather_ids(operation.input_shape, params)

    def _gather_ids(self, shape, params, ids=None):
        if ids is None:
            ids = {}
        for member_name, member_shape in shape.members.items():
            if member_shape.metadata.get('endpointdiscoveryid'):
                ids[member_name] = params[member_name]
            elif member_shape.type_name == 'structure' and member_name in params:
                self._gather_ids(member_shape, params[member_name], ids)
        return ids