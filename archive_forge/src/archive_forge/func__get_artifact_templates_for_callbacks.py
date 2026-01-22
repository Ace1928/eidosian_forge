import logging
import os
from typing import Collection, List, Optional, Type, Union, TYPE_CHECKING
from ray.tune.callback import Callback, CallbackList
from ray.tune.logger import (
def _get_artifact_templates_for_callbacks(callbacks: Union[List[Callback], List[Type[Callback]], CallbackList]) -> List[str]:
    templates = []
    for callback in callbacks:
        templates += list(callback._SAVED_FILE_TEMPLATES)
    return templates