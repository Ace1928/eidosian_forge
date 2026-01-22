import functools
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView, concat
def _list_morph_files_by(self, tag, values, map=None):
    fileids = self.fileids()
    ret_fileids = set()
    for f in fileids:
        fp = self.abspath(f).replace('morph.xml', 'header.xml')
        values_list = self._get_tag(fp, tag)
        for value in values_list:
            if map is not None:
                value = map(value)
            if value in values:
                ret_fileids.add(f)
    return list(ret_fileids)