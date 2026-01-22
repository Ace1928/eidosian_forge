import re
from collections import defaultdict
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import ElementTree, XMLCorpusReader
from nltk.util import LazyConcatenation, LazyMap, flatten
def _get_participants(self, fileid):

    def dictOfDicts():
        return defaultdict(dictOfDicts)
    xmldoc = ElementTree.parse(fileid).getroot()
    pat = dictOfDicts()
    for participant in xmldoc.findall(f'.//{{{NS}}}Participants/{{{NS}}}participant'):
        for key, value in participant.items():
            pat[participant.get('id')][key] = value
    return pat