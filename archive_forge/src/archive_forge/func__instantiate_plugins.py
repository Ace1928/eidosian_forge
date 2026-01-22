from __future__ import annotations
import collections.abc as collections_abc
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union
from urllib.parse import parse_qsl
from urllib.parse import quote
from urllib.parse import quote_plus
from urllib.parse import unquote
from .interfaces import Dialect
from .. import exc
from .. import util
from ..dialects import plugins
from ..dialects import registry
def _instantiate_plugins(self, kwargs: Mapping[str, Any]) -> Tuple[URL, List[Any], Dict[str, Any]]:
    plugin_names = util.to_list(self.query.get('plugin', ()))
    plugin_names += kwargs.get('plugins', [])
    kwargs = dict(kwargs)
    loaded_plugins = [plugins.load(plugin_name)(self, kwargs) for plugin_name in plugin_names]
    u = self.difference_update_query(['plugin', 'plugins'])
    for plugin in loaded_plugins:
        new_u = plugin.update_url(u)
        if new_u is not None:
            u = new_u
    kwargs.pop('plugins', None)
    return (u, loaded_plugins, kwargs)