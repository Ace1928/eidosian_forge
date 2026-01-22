import re
from nltk.corpus.reader.api import *
from nltk.tokenize import *
def _read_review_block(self, stream):
    while True:
        line = stream.readline()
        if not line:
            return []
        title_match = re.match(TITLE, line)
        if title_match:
            review = Review(title=title_match.group(1).strip())
            break
    while True:
        oldpos = stream.tell()
        line = stream.readline()
        if not line:
            return [review]
        if re.match(TITLE, line):
            stream.seek(oldpos)
            return [review]
        feats = re.findall(FEATURES, line)
        notes = re.findall(NOTES, line)
        sent = re.findall(SENT, line)
        if sent:
            sent = self._word_tokenizer.tokenize(sent[0])
        review_line = ReviewLine(sent=sent, features=feats, notes=notes)
        review.add_line(review_line)