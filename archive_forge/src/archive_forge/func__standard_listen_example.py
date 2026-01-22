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
def _standard_listen_example(dispatch_collection: _ClsLevelDispatch[_ET], sample_target: Any, fn: _ListenerFnType) -> str:
    example_kw_arg = _indent('\n'.join(("%(arg)s = kw['%(arg)s']" % {'arg': arg} for arg in dispatch_collection.arg_names[0:2])), '    ')
    if dispatch_collection.legacy_signatures:
        current_since = max((since for since, args, conv in dispatch_collection.legacy_signatures))
    else:
        current_since = None
    text = 'from sqlalchemy import event\n\n\n@event.listens_for(%(sample_target)s, \'%(event_name)s\')\ndef receive_%(event_name)s(%(named_event_arguments)s%(has_kw_arguments)s):\n    "listen for the \'%(event_name)s\' event"\n\n    # ... (event handling logic) ...\n'
    text %= {'current_since': ' (arguments as of %s)' % current_since if current_since else '', 'event_name': fn.__name__, 'has_kw_arguments': ', **kw' if dispatch_collection.has_kw else '', 'named_event_arguments': ', '.join(dispatch_collection.arg_names), 'example_kw_arg': example_kw_arg, 'sample_target': sample_target}
    return text