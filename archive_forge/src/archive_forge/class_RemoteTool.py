import base64
import importlib
import inspect
import io
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import create_repo, hf_hub_download, metadata_update, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError, build_hf_headers, get_session
from ..dynamic_module_utils import custom_object_save, get_class_from_dynamic_module, get_imports
from ..image_utils import is_pil_image
from ..models.auto import AutoProcessor
from ..utils import (
from .agent_types import handle_agent_inputs, handle_agent_outputs
from {module_name} import {class_name}
class RemoteTool(Tool):
    """
    A [`Tool`] that will make requests to an inference endpoint.

    Args:
        endpoint_url (`str`, *optional*):
            The url of the endpoint to use.
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when
            running `huggingface-cli login` (stored in `~/.huggingface`).
        tool_class (`type`, *optional*):
            The corresponding `tool_class` if this is a remote version of an existing tool. Will help determine when
            the output should be converted to another type (like images).
    """

    def __init__(self, endpoint_url=None, token=None, tool_class=None):
        self.endpoint_url = endpoint_url
        self.client = EndpointClient(endpoint_url, token=token)
        self.tool_class = tool_class

    def prepare_inputs(self, *args, **kwargs):
        """
        Prepare the inputs received for the HTTP client sending data to the endpoint. Positional arguments will be
        matched with the signature of the `tool_class` if it was provided at instantation. Images will be encoded into
        bytes.

        You can override this method in your custom class of [`RemoteTool`].
        """
        inputs = kwargs.copy()
        if len(args) > 0:
            if self.tool_class is not None:
                if issubclass(self.tool_class, PipelineTool):
                    call_method = self.tool_class.encode
                else:
                    call_method = self.tool_class.__call__
                signature = inspect.signature(call_method).parameters
                parameters = [k for k, p in signature.items() if p.kind not in [inspect._ParameterKind.VAR_POSITIONAL, inspect._ParameterKind.VAR_KEYWORD]]
                if parameters[0] == 'self':
                    parameters = parameters[1:]
                if len(args) > len(parameters):
                    raise ValueError(f'{self.tool_class} only accepts {len(parameters)} arguments but {len(args)} were given.')
                for arg, name in zip(args, parameters):
                    inputs[name] = arg
            elif len(args) > 1:
                raise ValueError('A `RemoteTool` can only accept one positional input.')
            elif len(args) == 1:
                if is_pil_image(args[0]):
                    return {'inputs': self.client.encode_image(args[0])}
                return {'inputs': args[0]}
        for key, value in inputs.items():
            if is_pil_image(value):
                inputs[key] = self.client.encode_image(value)
        return {'inputs': inputs}

    def extract_outputs(self, outputs):
        """
        You can override this method in your custom class of [`RemoteTool`] to apply some custom post-processing of the
        outputs of the endpoint.
        """
        return outputs

    def __call__(self, *args, **kwargs):
        args, kwargs = handle_agent_inputs(*args, **kwargs)
        output_image = self.tool_class is not None and self.tool_class.outputs == ['image']
        inputs = self.prepare_inputs(*args, **kwargs)
        if isinstance(inputs, dict):
            outputs = self.client(**inputs, output_image=output_image)
        else:
            outputs = self.client(inputs, output_image=output_image)
        if isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], list):
            outputs = outputs[0]
        outputs = handle_agent_outputs(outputs, self.tool_class.outputs if self.tool_class is not None else None)
        return self.extract_outputs(outputs)