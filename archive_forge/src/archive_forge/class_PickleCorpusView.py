import bisect
import os
import pickle
import re
import tempfile
from functools import reduce
from xml.etree import ElementTree
from nltk.data import (
from nltk.internals import slice_bounds
from nltk.tokenize import wordpunct_tokenize
from nltk.util import AbstractLazySequence, LazyConcatenation, LazySubsequence
class PickleCorpusView(StreamBackedCorpusView):
    """
    A stream backed corpus view for corpus files that consist of
    sequences of serialized Python objects (serialized using
    ``pickle.dump``).  One use case for this class is to store the
    result of running feature detection on a corpus to disk.  This can
    be useful when performing feature detection is expensive (so we
    don't want to repeat it); but the corpus is too large to store in
    memory.  The following example illustrates this technique:

        >>> from nltk.corpus.reader.util import PickleCorpusView
        >>> from nltk.util import LazyMap
        >>> feature_corpus = LazyMap(detect_features, corpus) # doctest: +SKIP
        >>> PickleCorpusView.write(feature_corpus, some_fileid)  # doctest: +SKIP
        >>> pcv = PickleCorpusView(some_fileid) # doctest: +SKIP
    """
    BLOCK_SIZE = 100
    PROTOCOL = -1

    def __init__(self, fileid, delete_on_gc=False):
        """
        Create a new corpus view that reads the pickle corpus
        ``fileid``.

        :param delete_on_gc: If true, then ``fileid`` will be deleted
            whenever this object gets garbage-collected.
        """
        self._delete_on_gc = delete_on_gc
        StreamBackedCorpusView.__init__(self, fileid)

    def read_block(self, stream):
        result = []
        for i in range(self.BLOCK_SIZE):
            try:
                result.append(pickle.load(stream))
            except EOFError:
                break
        return result

    def __del__(self):
        """
        If ``delete_on_gc`` was set to true when this
        ``PickleCorpusView`` was created, then delete the corpus view's
        fileid.  (This method is called whenever a
        ``PickledCorpusView`` is garbage-collected.
        """
        if getattr(self, '_delete_on_gc'):
            if os.path.exists(self._fileid):
                try:
                    os.remove(self._fileid)
                except OSError:
                    pass
        self.__dict__.clear()

    @classmethod
    def write(cls, sequence, output_file):
        if isinstance(output_file, str):
            output_file = open(output_file, 'wb')
        for item in sequence:
            pickle.dump(item, output_file, cls.PROTOCOL)

    @classmethod
    def cache_to_tempfile(cls, sequence, delete_on_gc=True):
        """
        Write the given sequence to a temporary file as a pickle
        corpus; and then return a ``PickleCorpusView`` view for that
        temporary corpus file.

        :param delete_on_gc: If true, then the temporary file will be
            deleted whenever this object gets garbage-collected.
        """
        try:
            fd, output_file_name = tempfile.mkstemp('.pcv', 'nltk-')
            output_file = os.fdopen(fd, 'wb')
            cls.write(sequence, output_file)
            output_file.close()
            return PickleCorpusView(output_file_name, delete_on_gc)
        except OSError as e:
            raise ValueError('Error while creating temp file: %s' % e) from e