import copy
import functools
import gc
import inspect
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from .utils import (
def _remove_columns(self, feature: dict) -> dict:
    if not isinstance(feature, dict):
        return feature
    if not self.message_logged and self.logger and self.model_name:
        ignored_columns = list(set(feature.keys()) - set(self.signature_columns))
        if len(ignored_columns) > 0:
            dset_description = '' if self.description is None else f'in the {self.description} set'
            self.logger.info(f"The following columns {dset_description} don't have a corresponding argument in `{self.model_name}.forward` and have been ignored: {', '.join(ignored_columns)}. If {', '.join(ignored_columns)} are not expected by `{self.model_name}.forward`,  you can safely ignore this message.")
            self.message_logged = True
    return {k: v for k, v in feature.items() if k in self.signature_columns}