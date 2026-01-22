import os
import io
import json
import pandas as pd
import pyarrow
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import ray
from ray.air.constants import (
from ray.util.annotations import PublicAPI
import logging
def _repr(self, indent: int=0) -> str:
    """Construct the representation with specified number of space indent."""
    from ray.tune.result import AUTO_RESULT_KEYS
    from ray.tune.experimental.output import BLACKLISTED_KEYS
    shown_attributes = {k: getattr(self, k) for k in self._items_to_repr}
    if self.error:
        shown_attributes['error'] = type(self.error).__name__
    else:
        shown_attributes.pop('error')
    shown_attributes['filesystem'] = shown_attributes['filesystem'].type_name
    if self.metrics:
        exclude = set(AUTO_RESULT_KEYS)
        exclude.update(BLACKLISTED_KEYS)
        shown_attributes['metrics'] = {k: v for k, v in self.metrics.items() if k not in exclude}
    cls_indent = ' ' * indent
    kws_indent = ' ' * (indent + 2)
    kws = [f'{kws_indent}{key}={value!r}' for key, value in shown_attributes.items()]
    kws_repr = ',\n'.join(kws)
    return '{0}{1}(\n{2}\n{0})'.format(cls_indent, type(self).__name__, kws_repr)