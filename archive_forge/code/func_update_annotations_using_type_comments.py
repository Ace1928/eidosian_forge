from inspect import Parameter, Signature, getsource
from typing import Any, Dict, List, cast
import sphinx
from sphinx.application import Sphinx
from sphinx.locale import __
from sphinx.pycode.ast import ast
from sphinx.pycode.ast import parse as ast_parse
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import inspect, logging
def update_annotations_using_type_comments(app: Sphinx, obj: Any, bound_method: bool) -> None:
    """Update annotations info of *obj* using type_comments."""
    try:
        type_sig = get_type_comment(obj, bound_method)
        if type_sig:
            sig = inspect.signature(obj, bound_method)
            for param in sig.parameters.values():
                if param.name not in obj.__annotations__:
                    annotation = type_sig.parameters[param.name].annotation
                    if annotation is not Parameter.empty:
                        obj.__annotations__[param.name] = ast_unparse(annotation)
            if 'return' not in obj.__annotations__:
                obj.__annotations__['return'] = type_sig.return_annotation
    except KeyError as exc:
        logger.warning(__('Failed to update signature for %r: parameter not found: %s'), obj, exc)
    except NotImplementedError as exc:
        logger.warning(__('Failed to parse type_comment for %r: %s'), obj, exc)