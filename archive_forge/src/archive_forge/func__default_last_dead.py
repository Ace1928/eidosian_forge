from __future__ import annotations
import time
import typing as t
from traitlets import Bool, Dict, Float, Instance, Integer, default
from traitlets.config.configurable import LoggingConfigurable
@default('_last_dead')
def _default_last_dead(self) -> float:
    return time.time()