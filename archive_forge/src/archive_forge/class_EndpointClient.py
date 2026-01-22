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
class EndpointClient:

    def __init__(self, endpoint_url: str, token: Optional[str]=None):
        self.headers = {**build_hf_headers(token=token), 'Content-Type': 'application/json'}
        self.endpoint_url = endpoint_url

    @staticmethod
    def encode_image(image):
        _bytes = io.BytesIO()
        image.save(_bytes, format='PNG')
        b64 = base64.b64encode(_bytes.getvalue())
        return b64.decode('utf-8')

    @staticmethod
    def decode_image(raw_image):
        if not is_vision_available():
            raise ImportError('This tool returned an image but Pillow is not installed. Please install it (`pip install Pillow`).')
        from PIL import Image
        b64 = base64.b64decode(raw_image)
        _bytes = io.BytesIO(b64)
        return Image.open(_bytes)

    def __call__(self, inputs: Optional[Union[str, Dict, List[str], List[List[str]]]]=None, params: Optional[Dict]=None, data: Optional[bytes]=None, output_image: bool=False) -> Any:
        payload = {}
        if inputs:
            payload['inputs'] = inputs
        if params:
            payload['parameters'] = params
        response = get_session().post(self.endpoint_url, headers=self.headers, json=payload, data=data)
        if output_image:
            return self.decode_image(response.content)
        else:
            return response.json()