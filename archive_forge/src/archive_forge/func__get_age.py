import re
from collections import defaultdict
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import ElementTree, XMLCorpusReader
from nltk.util import LazyConcatenation, LazyMap, flatten
def _get_age(self, fileid, speaker, month):
    xmldoc = ElementTree.parse(fileid).getroot()
    for pat in xmldoc.findall(f'.//{{{NS}}}Participants/{{{NS}}}participant'):
        try:
            if pat.get('id') == speaker:
                age = pat.get('age')
                if month:
                    age = self.convert_age(age)
                return age
        except (TypeError, AttributeError) as e:
            return None