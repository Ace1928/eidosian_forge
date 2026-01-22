from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Literal, TypedDict
from xarray.core.utils import FrozenDict
class set_options:
    """
    Set options for xarray in a controlled context.

    Parameters
    ----------
    arithmetic_join : {"inner", "outer", "left", "right", "exact"}, default: "inner"
        DataArray/Dataset alignment in binary operations:

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.

    cmap_divergent : str or matplotlib.colors.Colormap, default: "RdBu_r"
        Colormap to use for divergent data plots. If string, must be
        matplotlib built-in colormap. Can also be a Colormap object
        (e.g. mpl.colormaps["magma"])
    cmap_sequential : str or matplotlib.colors.Colormap, default: "viridis"
        Colormap to use for nondivergent data plots. If string, must be
        matplotlib built-in colormap. Can also be a Colormap object
        (e.g. mpl.colormaps["magma"])
    display_expand_attrs : {"default", True, False}
        Whether to expand the attributes section for display of
        ``DataArray`` or ``Dataset`` objects. Can be

        * ``True`` : to always expand attrs
        * ``False`` : to always collapse attrs
        * ``default`` : to expand unless over a pre-defined limit
    display_expand_coords : {"default", True, False}
        Whether to expand the coordinates section for display of
        ``DataArray`` or ``Dataset`` objects. Can be

        * ``True`` : to always expand coordinates
        * ``False`` : to always collapse coordinates
        * ``default`` : to expand unless over a pre-defined limit
    display_expand_data : {"default", True, False}
        Whether to expand the data section for display of ``DataArray``
        objects. Can be

        * ``True`` : to always expand data
        * ``False`` : to always collapse data
        * ``default`` : to expand unless over a pre-defined limit
    display_expand_data_vars : {"default", True, False}
        Whether to expand the data variables section for display of
        ``Dataset`` objects. Can be

        * ``True`` : to always expand data variables
        * ``False`` : to always collapse data variables
        * ``default`` : to expand unless over a pre-defined limit
    display_expand_indexes : {"default", True, False}
        Whether to expand the indexes section for display of
        ``DataArray`` or ``Dataset``. Can be

        * ``True`` : to always expand indexes
        * ``False`` : to always collapse indexes
        * ``default`` : to expand unless over a pre-defined limit (always collapse for html style)
    display_max_rows : int, default: 12
        Maximum display rows.
    display_values_threshold : int, default: 200
        Total number of array elements which trigger summarization rather
        than full repr for variable data views (numpy arrays).
    display_style : {"text", "html"}, default: "html"
        Display style to use in jupyter for xarray objects.
    display_width : int, default: 80
        Maximum display width for ``repr`` on xarray objects.
    file_cache_maxsize : int, default: 128
        Maximum number of open files to hold in xarray's
        global least-recently-usage cached. This should be smaller than
        your system's per-process file descriptor limit, e.g.,
        ``ulimit -n`` on Linux.
    keep_attrs : {"default", True, False}
        Whether to keep attributes on xarray Datasets/dataarrays after
        operations. Can be

        * ``True`` : to always keep attrs
        * ``False`` : to always discard attrs
        * ``default`` : to use original logic that attrs should only
          be kept in unambiguous circumstances
    use_bottleneck : bool, default: True
        Whether to use ``bottleneck`` to accelerate 1D reductions and
        1D rolling reduction operations.
    use_flox : bool, default: True
        Whether to use ``numpy_groupies`` and `flox`` to
        accelerate groupby and resampling reductions.
    use_numbagg : bool, default: True
        Whether to use ``numbagg`` to accelerate reductions.
        Takes precedence over ``use_bottleneck`` when both are True.
    use_opt_einsum : bool, default: True
        Whether to use ``opt_einsum`` to accelerate dot products.
    warn_for_unclosed_files : bool, default: False
        Whether or not to issue a warning when unclosed files are
        deallocated. This is mostly useful for debugging.

    Examples
    --------
    It is possible to use ``set_options`` either as a context manager:

    >>> ds = xr.Dataset({"x": np.arange(1000)})
    >>> with xr.set_options(display_width=40):
    ...     print(ds)
    ...
    <xarray.Dataset> Size: 8kB
    Dimensions:  (x: 1000)
    Coordinates:
      * x        (x) int64 8kB 0 1 ... 999
    Data variables:
        *empty*

    Or to set global options:

    >>> xr.set_options(display_width=80)  # doctest: +ELLIPSIS
    <xarray.core.options.set_options object at 0x...>
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(f'argument name {k!r} is not in the set of valid options {set(OPTIONS)!r}')
            if k in _VALIDATORS and (not _VALIDATORS[k](v)):
                if k == 'arithmetic_join':
                    expected = f'Expected one of {_JOIN_OPTIONS!r}'
                elif k == 'display_style':
                    expected = f'Expected one of {_DISPLAY_OPTIONS!r}'
                else:
                    expected = ''
                raise ValueError(f'option {k!r} given an invalid value: {v!r}. ' + expected)
            self.old[k] = OPTIONS[k]
        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        for k, v in options_dict.items():
            if k in _SETTERS:
                _SETTERS[k](v)
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)