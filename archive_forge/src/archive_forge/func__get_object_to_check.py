import inspect
from pathlib import Path
from jedi.parser_utils import get_cached_code_lines
from jedi import settings
from jedi.cache import memoize_method
from jedi.inference import compiled
from jedi.file_io import FileIO
from jedi.inference.names import NameWrapper
from jedi.inference.base_value import ValueSet, ValueWrapper, NO_VALUES
from jedi.inference.value import ModuleValue
from jedi.inference.cache import inference_state_function_cache, \
from jedi.inference.compiled.access import ALLOWED_GETITEM_TYPES, get_api_type
from jedi.inference.gradual.conversion import to_stub
from jedi.inference.context import CompiledContext, CompiledModuleContext, \
def _get_object_to_check(python_object):
    """Check if inspect.getfile has a chance to find the source."""
    try:
        python_object = inspect.unwrap(python_object)
    except ValueError:
        pass
    if inspect.ismodule(python_object) or inspect.isclass(python_object) or inspect.ismethod(python_object) or inspect.isfunction(python_object) or inspect.istraceback(python_object) or inspect.isframe(python_object) or inspect.iscode(python_object):
        return python_object
    try:
        return python_object.__class__
    except AttributeError:
        raise TypeError