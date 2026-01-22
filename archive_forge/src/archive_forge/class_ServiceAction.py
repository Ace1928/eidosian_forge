import logging
from botocore import xform_name
from .params import create_request_parameters
from .response import RawHandler, ResourceHandler
from .model import Action
from boto3.docs.docstring import ActionDocstring
from boto3.utils import inject_attribute
class ServiceAction(object):
    """
    A class representing a callable action on a resource, for example
    ``sqs.get_queue_by_name(...)`` or ``s3.Bucket('foo').delete()``.
    The action may construct parameters from existing resource identifiers
    and may return either a raw response or a new resource instance.

    :type action_model: :py:class`~boto3.resources.model.Action`
    :param action_model: The action model.

    :type factory: ResourceFactory
    :param factory: The factory that created the resource class to which
                    this action is attached.

    :type service_context: :py:class:`~boto3.utils.ServiceContext`
    :param service_context: Context about the AWS service
    """

    def __init__(self, action_model, factory=None, service_context=None):
        self._action_model = action_model
        resource_response_model = action_model.resource
        if resource_response_model:
            self._response_handler = ResourceHandler(search_path=resource_response_model.path, factory=factory, resource_model=resource_response_model, service_context=service_context, operation_name=action_model.request.operation)
        else:
            self._response_handler = RawHandler(action_model.path)

    def __call__(self, parent, *args, **kwargs):
        """
        Perform the action's request operation after building operation
        parameters and build any defined resources from the response.

        :type parent: :py:class:`~boto3.resources.base.ServiceResource`
        :param parent: The resource instance to which this action is attached.
        :rtype: dict or ServiceResource or list(ServiceResource)
        :return: The response, either as a raw dict or resource instance(s).
        """
        operation_name = xform_name(self._action_model.request.operation)
        params = create_request_parameters(parent, self._action_model.request)
        params.update(kwargs)
        logger.debug('Calling %s:%s with %r', parent.meta.service_name, operation_name, params)
        response = getattr(parent.meta.client, operation_name)(*args, **params)
        logger.debug('Response: %r', response)
        return self._response_handler(parent, params, response)