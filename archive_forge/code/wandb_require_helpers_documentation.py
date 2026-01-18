import os
from functools import wraps
from typing import Any, Callable, Dict, TypeVar, cast
Decorate functions to gate features with wandb.require.