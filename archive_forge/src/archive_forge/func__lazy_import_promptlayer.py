from __future__ import annotations
import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
from langchain_core.outputs import (
def _lazy_import_promptlayer() -> promptlayer:
    """Lazy import promptlayer to avoid circular imports."""
    try:
        import promptlayer
    except ImportError:
        raise ImportError('The PromptLayerCallbackHandler requires the promptlayer package.  Please install it with `pip install promptlayer`.')
    return promptlayer