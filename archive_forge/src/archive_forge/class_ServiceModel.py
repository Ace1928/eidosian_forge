from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
class ServiceModel:
    """

    :ivar service_description: The parsed service description dictionary.

    """

    def __init__(self, service_description, service_name=None):
        """

        :type service_description: dict
        :param service_description: The service description model.  This value
            is obtained from a botocore.loader.Loader, or from directly loading
            the file yourself::

                service_description = json.load(
                    open('/path/to/service-description-model.json'))
                model = ServiceModel(service_description)

        :type service_name: str
        :param service_name: The name of the service.  Normally this is
            the endpoint prefix defined in the service_description.  However,
            you can override this value to provide a more convenient name.
            This is done in a few places in botocore (ses instead of email,
            emr instead of elasticmapreduce).  If this value is not provided,
            it will default to the endpointPrefix defined in the model.

        """
        self._service_description = service_description
        self.metadata = service_description.get('metadata', {})
        self._shape_resolver = ShapeResolver(service_description.get('shapes', {}))
        self._signature_version = NOT_SET
        self._service_name = service_name
        self._instance_cache = {}

    def shape_for(self, shape_name, member_traits=None):
        return self._shape_resolver.get_shape_by_name(shape_name, member_traits)

    def shape_for_error_code(self, error_code):
        return self._error_code_cache.get(error_code, None)

    @CachedProperty
    def _error_code_cache(self):
        error_code_cache = {}
        for error_shape in self.error_shapes:
            code = error_shape.error_code
            error_code_cache[code] = error_shape
        return error_code_cache

    def resolve_shape_ref(self, shape_ref):
        return self._shape_resolver.resolve_shape_ref(shape_ref)

    @CachedProperty
    def shape_names(self):
        return list(self._service_description.get('shapes', {}))

    @CachedProperty
    def error_shapes(self):
        error_shapes = []
        for shape_name in self.shape_names:
            error_shape = self.shape_for(shape_name)
            if error_shape.metadata.get('exception', False):
                error_shapes.append(error_shape)
        return error_shapes

    @instance_cache
    def operation_model(self, operation_name):
        try:
            model = self._service_description['operations'][operation_name]
        except KeyError:
            raise OperationNotFoundError(operation_name)
        return OperationModel(model, self, operation_name)

    @CachedProperty
    def documentation(self):
        return self._service_description.get('documentation', '')

    @CachedProperty
    def operation_names(self):
        return list(self._service_description.get('operations', []))

    @CachedProperty
    def service_name(self):
        """The name of the service.

        This defaults to the endpointPrefix defined in the service model.
        However, this value can be overriden when a ``ServiceModel`` is
        created.  If a service_name was not provided when the ``ServiceModel``
        was created and if there is no endpointPrefix defined in the
        service model, then an ``UndefinedModelAttributeError`` exception
        will be raised.

        """
        if self._service_name is not None:
            return self._service_name
        else:
            return self.endpoint_prefix

    @CachedProperty
    def service_id(self):
        try:
            return ServiceId(self._get_metadata_property('serviceId'))
        except UndefinedModelAttributeError:
            raise MissingServiceIdError(service_name=self._service_name)

    @CachedProperty
    def signing_name(self):
        """The name to use when computing signatures.

        If the model does not define a signing name, this
        value will be the endpoint prefix defined in the model.
        """
        signing_name = self.metadata.get('signingName')
        if signing_name is None:
            signing_name = self.endpoint_prefix
        return signing_name

    @CachedProperty
    def api_version(self):
        return self._get_metadata_property('apiVersion')

    @CachedProperty
    def protocol(self):
        return self._get_metadata_property('protocol')

    @CachedProperty
    def endpoint_prefix(self):
        return self._get_metadata_property('endpointPrefix')

    @CachedProperty
    def endpoint_discovery_operation(self):
        for operation in self.operation_names:
            model = self.operation_model(operation)
            if model.is_endpoint_discovery_operation:
                return model

    @CachedProperty
    def endpoint_discovery_required(self):
        for operation in self.operation_names:
            model = self.operation_model(operation)
            if model.endpoint_discovery is not None and model.endpoint_discovery.get('required'):
                return True
        return False

    @CachedProperty
    def client_context_parameters(self):
        params = self._service_description.get('clientContextParams', {})
        return [ClientContextParameter(name=param_name, type=param_val['type'], documentation=param_val['documentation']) for param_name, param_val in params.items()]

    def _get_metadata_property(self, name):
        try:
            return self.metadata[name]
        except KeyError:
            raise UndefinedModelAttributeError(f'"{name}" not defined in the metadata of the model: {self}')

    @property
    def signature_version(self):
        if self._signature_version is NOT_SET:
            signature_version = self.metadata.get('signatureVersion')
            self._signature_version = signature_version
        return self._signature_version

    @signature_version.setter
    def signature_version(self, value):
        self._signature_version = value

    def __repr__(self):
        return f'{self.__class__.__name__}({self.service_name})'