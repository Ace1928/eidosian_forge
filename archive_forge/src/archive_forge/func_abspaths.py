import os
import re
from collections import defaultdict
from itertools import chain
from nltk.corpus.reader.util import *
from nltk.data import FileSystemPathPointer, PathPointer, ZipFilePathPointer
def abspaths(self, fileids=None, include_encoding=False, include_fileid=False):
    """
        Return a list of the absolute paths for all fileids in this corpus;
        or for the given list of fileids, if specified.

        :type fileids: None or str or list
        :param fileids: Specifies the set of fileids for which paths should
            be returned.  Can be None, for all fileids; a list of
            file identifiers, for a specified set of fileids; or a single
            file identifier, for a single file.  Note that the return
            value is always a list of paths, even if ``fileids`` is a
            single file identifier.

        :param include_encoding: If true, then return a list of
            ``(path_pointer, encoding)`` tuples.

        :rtype: list(PathPointer)
        """
    if fileids is None:
        fileids = self._fileids
    elif isinstance(fileids, str):
        fileids = [fileids]
    paths = [self._root.join(f) for f in fileids]
    if include_encoding and include_fileid:
        return list(zip(paths, [self.encoding(f) for f in fileids], fileids))
    elif include_fileid:
        return list(zip(paths, fileids))
    elif include_encoding:
        return list(zip(paths, [self.encoding(f) for f in fileids]))
    else:
        return paths