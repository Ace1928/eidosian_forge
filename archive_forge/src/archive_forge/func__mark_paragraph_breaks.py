import math
import re
from nltk.tokenize.api import TokenizerI
def _mark_paragraph_breaks(self, text):
    """Identifies indented text or line breaks as the beginning of
        paragraphs"""
    MIN_PARAGRAPH = 100
    pattern = re.compile('[ \t\r\x0c\x0b]*\n[ \t\r\x0c\x0b]*\n[ \t\r\x0c\x0b]*')
    matches = pattern.finditer(text)
    last_break = 0
    pbreaks = [0]
    for pb in matches:
        if pb.start() - last_break < MIN_PARAGRAPH:
            continue
        else:
            pbreaks.append(pb.start())
            last_break = pb.start()
    return pbreaks