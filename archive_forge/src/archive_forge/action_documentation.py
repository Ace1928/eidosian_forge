import logging
from botocore import xform_name
from .params import create_request_parameters
from .response import RawHandler, ResourceHandler
from .model import Action
from boto3.docs.docstring import ActionDocstring
from boto3.utils import inject_attribute

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
        