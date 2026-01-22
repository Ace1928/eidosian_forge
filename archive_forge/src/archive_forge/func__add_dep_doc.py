from __future__ import annotations
import functools
import re
import typing as ty
import warnings
def _add_dep_doc(old_doc: str, dep_doc: str, setup: str='', cleanup: str='') -> str:
    """Add deprecation message `dep_doc` to docstring in `old_doc`

    Parameters
    ----------
    old_doc : str
        Docstring from some object.
    dep_doc : str
        Deprecation warning to add to top of docstring, after initial line.
    setup : str, optional
        Doctest setup text
    cleanup : str, optional
        Doctest teardown text

    Returns
    -------
    new_doc : str
        `old_doc` with `dep_doc` inserted after any first lines of docstring.
    """
    dep_doc = _ensure_cr(dep_doc)
    if not old_doc:
        return dep_doc
    old_doc = _ensure_cr(old_doc)
    old_lines = old_doc.splitlines()
    new_lines = []
    for line_no, line in enumerate(old_lines):
        if line.strip():
            new_lines.append(line)
        else:
            break
    next_line = line_no + 1
    if next_line >= len(old_lines):
        return old_doc + '\n' + dep_doc
    leading_white = _LEADING_WHITE.match(old_lines[next_line])
    assert leading_white is not None
    indent = leading_white.group()
    setup_lines = [indent + L for L in setup.splitlines()]
    dep_lines = [indent + L for L in [''] + dep_doc.splitlines() + ['']]
    cleanup_lines = [indent + L for L in cleanup.splitlines()]
    return '\n'.join(new_lines + dep_lines + setup_lines + old_lines[next_line:] + cleanup_lines + [''])