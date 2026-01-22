import os
import re
from collections import defaultdict
from itertools import chain
from nltk.corpus.reader.util import *
from nltk.data import FileSystemPathPointer, PathPointer, ZipFilePathPointer
def _read_tagged_sent_block(self, stream, tagset=None):
    return list(filter(None, [self._tag(t, tagset) for t in self._read_block(stream)]))