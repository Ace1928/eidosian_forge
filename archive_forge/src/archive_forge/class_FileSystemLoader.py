import importlib.util
import os
import posixpath
import sys
import typing as t
import weakref
import zipimport
from collections import abc
from hashlib import sha1
from importlib import import_module
from types import ModuleType
from .exceptions import TemplateNotFound
from .utils import internalcode
class FileSystemLoader(BaseLoader):
    """Load templates from a directory in the file system.

    The path can be relative or absolute. Relative paths are relative to
    the current working directory.

    .. code-block:: python

        loader = FileSystemLoader("templates")

    A list of paths can be given. The directories will be searched in
    order, stopping at the first matching template.

    .. code-block:: python

        loader = FileSystemLoader(["/override/templates", "/default/templates"])

    :param searchpath: A path, or list of paths, to the directory that
        contains the templates.
    :param encoding: Use this encoding to read the text from template
        files.
    :param followlinks: Follow symbolic links in the path.

    .. versionchanged:: 2.8
        Added the ``followlinks`` parameter.
    """

    def __init__(self, searchpath: t.Union[str, os.PathLike, t.Sequence[t.Union[str, os.PathLike]]], encoding: str='utf-8', followlinks: bool=False) -> None:
        if not isinstance(searchpath, abc.Iterable) or isinstance(searchpath, str):
            searchpath = [searchpath]
        self.searchpath = [os.fspath(p) for p in searchpath]
        self.encoding = encoding
        self.followlinks = followlinks

    def get_source(self, environment: 'Environment', template: str) -> t.Tuple[str, str, t.Callable[[], bool]]:
        pieces = split_template_path(template)
        for searchpath in self.searchpath:
            filename = posixpath.join(searchpath, *pieces)
            if os.path.isfile(filename):
                break
        else:
            raise TemplateNotFound(template)
        with open(filename, encoding=self.encoding) as f:
            contents = f.read()
        mtime = os.path.getmtime(filename)

        def uptodate() -> bool:
            try:
                return os.path.getmtime(filename) == mtime
            except OSError:
                return False
        return (contents, os.path.normpath(filename), uptodate)

    def list_templates(self) -> t.List[str]:
        found = set()
        for searchpath in self.searchpath:
            walk_dir = os.walk(searchpath, followlinks=self.followlinks)
            for dirpath, _, filenames in walk_dir:
                for filename in filenames:
                    template = os.path.join(dirpath, filename)[len(searchpath):].strip(os.path.sep).replace(os.path.sep, '/')
                    if template[:2] == './':
                        template = template[2:]
                    if template not in found:
                        found.add(template)
        return sorted(found)