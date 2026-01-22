import logging
from botocore import xform_name
from .params import create_request_parameters
from .response import RawHandler, ResourceHandler
from .model import Action
from boto3.docs.docstring import ActionDocstring
from boto3.utils import inject_attribute
class CustomModeledAction(object):
    """A custom, modeled action to inject into a resource."""

    def __init__(self, action_name, action_model, function, event_emitter):
        """
        :type action_name: str
        :param action_name: The name of the action to inject, e.g.
            'delete_tags'

        :type action_model: dict
        :param action_model: A JSON definition of the action, as if it were
            part of the resource model.

        :type function: function
        :param function: The function to perform when the action is called.
            The first argument should be 'self', which will be the resource
            the function is to be called on.

        :type event_emitter: :py:class:`botocore.hooks.BaseEventHooks`
        :param event_emitter: The session event emitter.
        """
        self.name = action_name
        self.model = action_model
        self.function = function
        self.emitter = event_emitter

    def inject(self, class_attributes, service_context, event_name, **kwargs):
        resource_name = event_name.rsplit('.')[-1]
        action = Action(self.name, self.model, {})
        self.function.__name__ = self.name
        self.function.__doc__ = ActionDocstring(resource_name=resource_name, event_emitter=self.emitter, action_model=action, service_model=service_context.service_model, include_signature=False)
        inject_attribute(class_attributes, self.name, self.function)