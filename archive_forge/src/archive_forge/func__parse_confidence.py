import re
from io import StringIO
from Bio.Phylo import Newick
def _parse_confidence(text):
    if text.isdigit():
        return int(text)
    try:
        return float(text)
    except ValueError:
        return None