import asyncio
import os
from typing import Awaitable, Tuple, Type, TypeVar, Union
from typing import Dict as TypeDict
from typing import List as TypeList
from pathlib import Path
from traitlets.traitlets import Dict, Float, List, default
from nbclient.util import ensure_async
import re
from .notebook_renderer import NotebookRenderer
from .utils import ENV_VARIABLE
@default('kernel_pools_config')
def _kernel_pools_config(self):
    return {'default': {'pool_size': max(default_pool_size, 0), 'kernel_env_variables': self.default_env_variables}}