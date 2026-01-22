from io import BytesIO
from typing import Callable, Dict, List, Tuple
from .. import errors, osutils, registry
def _get_registered_names():
    """Get the list of names with filters registered."""
    return filter_stacks_registry.keys()