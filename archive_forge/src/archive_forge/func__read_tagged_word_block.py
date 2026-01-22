import os
import re
from collections import defaultdict
from itertools import chain
from nltk.corpus.reader.util import *
from nltk.data import FileSystemPathPointer, PathPointer, ZipFilePathPointer
def _read_tagged_word_block(self, stream, tagset=None):
    return list(chain.from_iterable(self._read_tagged_sent_block(stream, tagset)))