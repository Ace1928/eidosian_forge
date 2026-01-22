import jmespath
from botocore import xform_name
from .params import get_data_member
class ResourceHandler(object):
    """
    Creates a new resource or list of new resources from the low-level
    response based on the given response resource definition.

    :type search_path: string
    :param search_path: JMESPath expression to search in the response

    :type factory: ResourceFactory
    :param factory: The factory that created the resource class to which
                    this action is attached.

    :type resource_model: :py:class:`~boto3.resources.model.ResponseResource`
    :param resource_model: Response resource model.

    :type service_context: :py:class:`~boto3.utils.ServiceContext`
    :param service_context: Context about the AWS service

    :type operation_name: string
    :param operation_name: Name of the underlying service operation, if it
                           exists.

    :rtype: ServiceResource or list
    :return: New resource instance(s).
    """

    def __init__(self, search_path, factory, resource_model, service_context, operation_name=None):
        self.search_path = search_path
        self.factory = factory
        self.resource_model = resource_model
        self.operation_name = operation_name
        self.service_context = service_context

    def __call__(self, parent, params, response):
        """
        :type parent: ServiceResource
        :param parent: The resource instance to which this action is attached.
        :type params: dict
        :param params: Request parameters sent to the service.
        :type response: dict
        :param response: Low-level operation response.
        """
        resource_name = self.resource_model.type
        json_definition = self.service_context.resource_json_definitions.get(resource_name)
        resource_cls = self.factory.load_from_definition(resource_name=resource_name, single_resource_json_definition=json_definition, service_context=self.service_context)
        raw_response = response
        search_response = None
        if self.search_path:
            search_response = jmespath.search(self.search_path, raw_response)
        identifiers = dict(build_identifiers(self.resource_model.identifiers, parent, params, raw_response))
        plural = [v for v in identifiers.values() if isinstance(v, list)]
        if plural:
            response = []
            for i in range(len(plural[0])):
                response_item = None
                if search_response:
                    response_item = search_response[i]
                response.append(self.handle_response_item(resource_cls, parent, identifiers, response_item))
        elif all_not_none(identifiers.values()):
            response = self.handle_response_item(resource_cls, parent, identifiers, search_response)
        else:
            response = None
            if self.operation_name is not None:
                response = build_empty_response(self.search_path, self.operation_name, self.service_context.service_model)
        return response

    def handle_response_item(self, resource_cls, parent, identifiers, resource_data):
        """
        Handles the creation of a single response item by setting
        parameters and creating the appropriate resource instance.

        :type resource_cls: ServiceResource subclass
        :param resource_cls: The resource class to instantiate.
        :type parent: ServiceResource
        :param parent: The resource instance to which this action is attached.
        :type identifiers: dict
        :param identifiers: Map of identifier names to value or values.
        :type resource_data: dict or None
        :param resource_data: Data for resource attributes.
        :rtype: ServiceResource
        :return: New resource instance.
        """
        kwargs = {'client': parent.meta.client}
        for name, value in identifiers.items():
            if isinstance(value, list):
                value = value.pop(0)
            kwargs[name] = value
        resource = resource_cls(**kwargs)
        if resource_data is not None:
            resource.meta.data = resource_data
        return resource