from typing import Any, Dict, List, Optional, Generic, TypeVar, cast
from types import TracebackType
from importlib.metadata import entry_points
from toolz import curry
def _enable(self, name: str, **options) -> None:
    if name not in self._plugins:
        try:
            ep, = [ep for ep in importlib_metadata_get(self.entry_point_group) if ep.name == name]
        except ValueError as err:
            if name in self.entrypoint_err_messages:
                raise ValueError(self.entrypoint_err_messages[name]) from err
            else:
                raise NoSuchEntryPoint(self.entry_point_group, name) from err
        value = cast(PluginType, ep.load())
        self.register(name, value)
    self._active_name = name
    self._active = self._plugins[name]
    for key in set(options.keys()) & set(self._global_settings.keys()):
        self._global_settings[key] = options.pop(key)
    self._options = options