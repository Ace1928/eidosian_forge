from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Literal, TypedDict
from xarray.core.utils import FrozenDict
class T_Options(TypedDict):
    arithmetic_broadcast: bool
    arithmetic_join: Literal['inner', 'outer', 'left', 'right', 'exact']
    cmap_divergent: str | Colormap
    cmap_sequential: str | Colormap
    display_max_rows: int
    display_values_threshold: int
    display_style: Literal['text', 'html']
    display_width: int
    display_expand_attrs: Literal['default', True, False]
    display_expand_coords: Literal['default', True, False]
    display_expand_data_vars: Literal['default', True, False]
    display_expand_data: Literal['default', True, False]
    display_expand_groups: Literal['default', True, False]
    display_expand_indexes: Literal['default', True, False]
    display_default_indexes: Literal['default', True, False]
    enable_cftimeindex: bool
    file_cache_maxsize: int
    keep_attrs: Literal['default', True, False]
    warn_for_unclosed_files: bool
    use_bottleneck: bool
    use_flox: bool
    use_numbagg: bool
    use_opt_einsum: bool