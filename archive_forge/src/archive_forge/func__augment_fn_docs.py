from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from .registry import _ET
from .registry import _ListenerFnType
from .. import util
from ..util.compat import FullArgSpec
def _augment_fn_docs(dispatch_collection: _ClsLevelDispatch[_ET], parent_dispatch_cls: Type[_HasEventsDispatch[_ET]], fn: _ListenerFnType) -> str:
    header = '.. container:: event_signatures\n\n     Example argument forms::\n\n'
    sample_target = getattr(parent_dispatch_cls, '_target_class_doc', 'obj')
    text = header + _indent(_standard_listen_example(dispatch_collection, sample_target, fn), ' ' * 8)
    if dispatch_collection.legacy_signatures:
        text += _indent(_legacy_listen_examples(dispatch_collection, sample_target, fn), ' ' * 8)
        text += _version_signature_changes(parent_dispatch_cls, dispatch_collection)
    return util.inject_docstring_text(fn.__doc__, text, 1)