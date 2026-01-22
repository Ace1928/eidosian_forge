import os
import os.path
import posixpath
import stat
from .compat import Collection, Iterable, string_types, unicode
def _normalize_entries(entries, separators=None):
    """
	Normalizes the entry paths to use the POSIX path separator.

	*entries* (:class:`~collections.abc.Iterable` of :class:`.TreeEntry`)
	contains the entries to be normalized.

	*separators* (:class:`~collections.abc.Collection` of :class:`str`; or
	:data:`None`) optionally contains the path separators to normalize.
	See :func:`normalize_file` for more information.

	Returns a :class:`dict` mapping the each normalized file path (:class:`str`)
	to the entry (:class:`.TreeEntry`)
	"""
    norm_files = {}
    for entry in entries:
        norm_files[normalize_file(entry.path, separators=separators)] = entry
    return norm_files