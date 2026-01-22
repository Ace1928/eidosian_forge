from .basedatatypes import Undefined
from .optional_imports import get_module
def _js_to_py(v, widget_manager):
    """
    Javascript -> Python ipywidget deserializer

    Parameters
    ----------
    v
        Object to be deserialized
    widget_manager
        ipywidget widget_manager (unused)

    Returns
    -------
    any
        Deserialized object for use by the Python side of the library
    """
    if isinstance(v, dict):
        return {k: _js_to_py(v, widget_manager) for k, v in v.items()}
    elif isinstance(v, (list, tuple)):
        return [_js_to_py(v, widget_manager) for v in v]
    elif isinstance(v, str) and v == '_undefined_':
        return Undefined
    else:
        return v