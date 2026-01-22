import re
from nltk.corpus.reader import CorpusReader
def _parse_src_file(self):
    lines = self.open(self._fileids[0]).read().splitlines()
    lines = filter(lambda x: not re.search('^\\s*#', x), lines)
    for i, line in enumerate(lines):
        fields = [field.strip() for field in re.split('\\t+', line)]
        try:
            pos, offset, pos_score, neg_score, synset_terms, gloss = fields
        except BaseException as e:
            raise ValueError(f'Line {i} formatted incorrectly: {line}\n') from e
        if pos and offset:
            offset = int(offset)
            self._db[pos, offset] = (float(pos_score), float(neg_score))