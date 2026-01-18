from __future__ import annotations
import os
import pathlib
import typing as ty
def types_filenames(template_fname: FileSpec, types_exts: ty.Sequence[ExtensionSpec], trailing_suffixes: ty.Sequence[str]=('.gz', '.bz2'), enforce_extensions: bool=True, match_case: bool=False) -> dict[str, str]:
    """Return filenames with standard extensions from template name

    The typical case is returning image and header filenames for an
    Analyze image, that expects an 'image' file type with extension ``.img``,
    and a 'header' file type, with extension ``.hdr``.

    Parameters
    ----------
    template_fname : str or os.PathLike
       template filename from which to construct output dict of
       filenames, with given `types_exts` type to extension mapping.  If
       ``self.enforce_extensions`` is True, then filename must have one
       of the defined extensions from the types list.  If
       ``self.enforce_extensions`` is False, then the other filenames
       are guessed at by adding extensions to the base filename.
       Ignored suffixes (from `trailing_suffixes`) append themselves to
       the end of all the filenames.
    types_exts : sequence of sequences
       sequence of (name, extension) str sequences defining type to
       extension mapping.
    trailing_suffixes : sequence of strings, optional
        suffixes that should be ignored when looking for
        extensions - default is ``('.gz', '.bz2')``
    enforce_extensions : {True, False}, optional
        If True, raise an error when attempting to set value to
        type which has the wrong extension
    match_case : bool, optional
       If True, match case of extensions and trailing suffixes when
       searching in `template_fname`, otherwise do case-insensitive
       match.

    Returns
    -------
    types_fnames : dict
       dict with types as keys, and generated filenames as values.  The
       types are given by the first elements of the tuples in
       `types_exts`.

    Examples
    --------
    >>> types_exts = (('t1','.ext1'),('t2', '.ext2'))
    >>> tfns = types_filenames('/path/test.ext1', types_exts)
    >>> tfns == {'t1': '/path/test.ext1', 't2': '/path/test.ext2'}
    True

    Bare file roots without extensions get them added

    >>> tfns = types_filenames('/path/test', types_exts)
    >>> tfns == {'t1': '/path/test.ext1', 't2': '/path/test.ext2'}
    True

    With enforce_extensions == False, allow first type to have any
    extension.

    >>> tfns = types_filenames('/path/test.funny', types_exts,
    ...                        enforce_extensions=False)
    >>> tfns == {'t1': '/path/test.funny', 't2': '/path/test.ext2'}
    True
    """
    template_fname = _stringify_path(template_fname)
    if not isinstance(template_fname, str):
        raise TypesFilenamesError('Need file name as input to set_filenames')
    if template_fname.endswith('.'):
        template_fname = template_fname[:-1]
    filename, found_ext, ignored, guessed_name = parse_filename(template_fname, types_exts, trailing_suffixes, match_case)
    direct_set_name = None
    if enforce_extensions:
        if guessed_name is None:
            if found_ext:
                raise TypesFilenamesError(f'File extension "{found_ext}" was not in expected list: {[e for t, e in types_exts]}')
            elif ignored:
                raise TypesFilenamesError(f'Confusing ignored suffix {ignored} without extension')
    elif found_ext or ignored:
        direct_set_name = types_exts[0][0]
    tfns = {}
    proc_ext: ty.Callable[[str], str] = lambda s: s
    if found_ext:
        if found_ext == found_ext.upper():
            proc_ext = str.upper
        elif found_ext == found_ext.lower():
            proc_ext = str.lower
    for name, ext in types_exts:
        if name == direct_set_name:
            tfns[name] = template_fname
            continue
        fname = filename
        if ext:
            fname += proc_ext(ext)
        if ignored:
            fname += ignored
        tfns[name] = fname
    return tfns