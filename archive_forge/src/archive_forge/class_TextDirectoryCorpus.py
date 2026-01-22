from __future__ import with_statement
import logging
import os
import random
import re
import sys
from gensim import interfaces, utils
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import (
from gensim.utils import deaccent, simple_tokenize
from smart_open import open
class TextDirectoryCorpus(TextCorpus):
    """Read documents recursively from a directory.
    Each file/line (depends on `lines_are_documents`) is interpreted as a plain text document.

    """

    def __init__(self, input, dictionary=None, metadata=False, min_depth=0, max_depth=None, pattern=None, exclude_pattern=None, lines_are_documents=False, encoding='utf-8', **kwargs):
        """

        Parameters
        ----------
        input : str
            Path to input file/folder.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            If a dictionary is provided, it will not be updated with the given corpus on initialization.
            If None - new dictionary will be built for the given corpus.
            If `input` is None, the dictionary will remain uninitialized.
        metadata : bool, optional
            If True - yield metadata with each document.
        min_depth : int, optional
            Minimum depth in directory tree at which to begin searching for files.
        max_depth : int, optional
            Max depth in directory tree at which files will no longer be considered.
            If None - not limited.
        pattern : str, optional
            Regex to use for file name inclusion, all those files *not* matching this pattern will be ignored.
        exclude_pattern : str, optional
            Regex to use for file name exclusion, all files matching this pattern will be ignored.
        lines_are_documents : bool, optional
            If True - each line is considered a document, otherwise - each file is one document.
        encoding : str, optional
            Encoding used to read the specified file or files in the specified directory.
        kwargs: keyword arguments passed through to the `TextCorpus` constructor.
            See :meth:`gemsim.corpora.textcorpus.TextCorpus.__init__` docstring for more details on these.

        """
        self._min_depth = min_depth
        self._max_depth = sys.maxsize if max_depth is None else max_depth
        self.pattern = pattern
        self.exclude_pattern = exclude_pattern
        self.lines_are_documents = lines_are_documents
        self.encoding = encoding
        super(TextDirectoryCorpus, self).__init__(input, dictionary, metadata, **kwargs)

    @property
    def lines_are_documents(self):
        return self._lines_are_documents

    @lines_are_documents.setter
    def lines_are_documents(self, lines_are_documents):
        self._lines_are_documents = lines_are_documents
        self.length = None

    @property
    def pattern(self):
        return self._pattern

    @pattern.setter
    def pattern(self, pattern):
        self._pattern = None if pattern is None else re.compile(pattern)
        self.length = None

    @property
    def exclude_pattern(self):
        return self._exclude_pattern

    @exclude_pattern.setter
    def exclude_pattern(self, pattern):
        self._exclude_pattern = None if pattern is None else re.compile(pattern)
        self.length = None

    @property
    def min_depth(self):
        return self._min_depth

    @min_depth.setter
    def min_depth(self, min_depth):
        self._min_depth = min_depth
        self.length = None

    @property
    def max_depth(self):
        return self._max_depth

    @max_depth.setter
    def max_depth(self, max_depth):
        self._max_depth = max_depth
        self.length = None

    def iter_filepaths(self):
        """Generate (lazily)  paths to each file in the directory structure within the specified range of depths.
        If a filename pattern to match was given, further filter to only those filenames that match.

        Yields
        ------
        str
            Path to file

        """
        for depth, dirpath, dirnames, filenames in walk(self.input):
            if self.min_depth <= depth <= self.max_depth:
                if self.pattern is not None:
                    filenames = (n for n in filenames if self.pattern.match(n) is not None)
                if self.exclude_pattern is not None:
                    filenames = (n for n in filenames if self.exclude_pattern.match(n) is None)
                for name in filenames:
                    yield os.path.join(dirpath, name)

    def getstream(self):
        """Generate documents from the underlying plain text collection (of one or more files).

        Yields
        ------
        str
            One document (if lines_are_documents - True), otherwise - each file is one document.

        """
        num_texts = 0
        for path in self.iter_filepaths():
            with open(path, 'rt', encoding=self.encoding) as f:
                if self.lines_are_documents:
                    for line in f:
                        yield line.strip()
                        num_texts += 1
                else:
                    yield f.read().strip()
                    num_texts += 1
        self.length = num_texts

    def __len__(self):
        """Get length of corpus.

        Returns
        -------
        int
            Length of corpus.

        """
        if self.length is None:
            self._cache_corpus_length()
        return self.length

    def _cache_corpus_length(self):
        """Calculate length of corpus and cache it to `self.length`."""
        if not self.lines_are_documents:
            self.length = sum((1 for _ in self.iter_filepaths()))
        else:
            self.length = sum((1 for _ in self.getstream()))