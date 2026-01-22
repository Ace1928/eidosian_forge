from __future__ import annotations
import os
import pathlib
import typing as ty
Split ``/pth/fname.ext.gz`` into ``/pth/fname, .ext, .gz``

    where ``.gz`` may be any of passed `addext` trailing suffixes.

    Parameters
    ----------
    filename : str or os.PathLike
       filename that may end in any or none of `addexts`
    match_case : bool, optional
       If True, match case of `addexts` and `filename`, otherwise do
       case-insensitive match.

    Returns
    -------
    froot : str
       Root of filename - e.g. ``/pth/fname`` in example above
    ext : str
       Extension, where extension is not in `addexts` - e.g. ``.ext`` in
       example above
    addext : str
       Any suffixes appearing in `addext` occurring at end of filename

    Examples
    --------
    >>> splitext_addext('fname.ext.gz')
    ('fname', '.ext', '.gz')
    >>> splitext_addext('fname.ext')
    ('fname', '.ext', '')
    >>> splitext_addext('fname.ext.foo', ('.foo', '.bar'))
    ('fname', '.ext', '.foo')
    