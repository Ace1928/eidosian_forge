import platform
from dataclasses import field
from enum import Enum
from typing import Dict, List, Optional, Union
from . import is_pydantic_available
from .doc import generate_doc_dataclass
class APIFeaturesManager:
    _SUPPORTED_TASKS = ['text-classification', 'token-classification', 'question-answering', 'image-classification']

    @staticmethod
    def check_supported_model_task_pair(model_type: str, task: str):
        model_type = model_type.lower()
        if model_type not in APIFeaturesManager._SUPPORTED_MODEL_TYPE:
            raise KeyError(f'{model_type} is not supported yet. Only {list(APIFeaturesManager._SUPPORTED_MODEL_TYPE.keys())} are supported. If you want to support {model_type} please propose a PR or open up an issue.')
        elif task not in APIFeaturesManager._SUPPORTED_MODEL_TYPE[model_type]:
            raise KeyError(f'{task} is not supported yet for model {model_type}. Only {APIFeaturesManager._SUPPORTED_MODEL_TYPE[model_type]} are supported. If you want to support {task} please propose a PR or open up an issue.')

    @staticmethod
    def check_supported_task(task: str):
        if task not in APIFeaturesManager._SUPPORTED_TASKS:
            raise KeyError(f'{task} is not supported yet. Only {APIFeaturesManager._SUPPORTED_TASKS} are supported. If you want to support {task} please propose a PR or open up an issue.')