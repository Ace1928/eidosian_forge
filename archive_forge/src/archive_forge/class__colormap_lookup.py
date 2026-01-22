from __future__ import annotations
import typing
from ._colormaps import ColorMap
from ._named_color_values import CRAYON, CSS4, SHORT, XKCD
class _colormap_lookup(dict[str, ColorMap]):
    """
    Lookup (by name) for all available colormaps
    """
    d: dict[str, ColorMap] = {}

    def _lazy_init(self):
        from ._colormaps._maps import _interpolated, _listed, _palette_interpolated, _segment_function, _segment_interpolated

        def _get(mod: ModuleType) -> dict[str, ColorMap]:
            return {name: getattr(mod, name) for name in mod.__all__}
        self.d = {**_get(_interpolated), **_get(_listed), **_get(_palette_interpolated), **_get(_segment_function), **_get(_segment_interpolated)}

    def __getitem__(self, name: str) -> ColorMap:
        if not self.d:
            self._lazy_init()
        try:
            return self.d[name]
        except KeyError as err:
            raise ValueError(f'Unknow colormap: {name}') from err