from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.parse import DependencyGraph
from nltk.tokenize import *
class DependencyCorpusView(StreamBackedCorpusView):
    _DOCSTART = '-DOCSTART- -DOCSTART- O\n'

    def __init__(self, corpus_file, tagged, group_by_sent, dependencies, chunk_types=None, encoding='utf8'):
        self._tagged = tagged
        self._dependencies = dependencies
        self._group_by_sent = group_by_sent
        self._chunk_types = chunk_types
        StreamBackedCorpusView.__init__(self, corpus_file, encoding=encoding)

    def read_block(self, stream):
        sent = read_blankline_block(stream)[0].strip()
        if sent.startswith(self._DOCSTART):
            sent = sent[len(self._DOCSTART):].lstrip()
        if not self._dependencies:
            lines = [line.split('\t') for line in sent.split('\n')]
            if len(lines[0]) == 3 or len(lines[0]) == 4:
                sent = [(line[0], line[1]) for line in lines]
            elif len(lines[0]) == 10:
                sent = [(line[1], line[4]) for line in lines]
            else:
                raise ValueError('Unexpected number of fields in dependency tree file')
            if not self._tagged:
                sent = [word for word, tag in sent]
        if self._group_by_sent:
            return [sent]
        else:
            return list(sent)