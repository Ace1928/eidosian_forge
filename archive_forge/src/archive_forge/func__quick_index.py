import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def _quick_index(self):
    """
        Initialize the indexes ``_lemma_to_class``,
        ``_wordnet_to_class``, and ``_class_to_fileid`` by scanning
        through the corpus fileids.  This doesn't do proper xml parsing,
        but is good enough to find everything in the standard VerbNet
        corpus -- and it runs about 30 times faster than xml parsing
        (with the python ElementTree; only 2-3 times faster
        if ElementTree uses the C implementation).
        """
    for fileid in self._fileids:
        vnclass = fileid[:-4]
        self._class_to_fileid[vnclass] = fileid
        self._shortid_to_longid[self.shortid(vnclass)] = vnclass
        with self.open(fileid) as fp:
            for m in self._INDEX_RE.finditer(fp.read()):
                groups = m.groups()
                if groups[0] is not None:
                    self._lemma_to_class[groups[0]].append(vnclass)
                    for wn in groups[1].split():
                        self._wordnet_to_class[wn].append(vnclass)
                elif groups[2] is not None:
                    self._class_to_fileid[groups[2]] = fileid
                    vnclass = groups[2]
                    self._shortid_to_longid[self.shortid(vnclass)] = vnclass
                else:
                    assert False, 'unexpected match condition'