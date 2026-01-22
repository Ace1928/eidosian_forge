import textwrap
from collections import defaultdict
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
@staticmethod
def _format_long_text(prefix, text):
    """Format the text as wrapped lines (PRIVATE)."""
    if text is None:
        return ''
    return textwrap.fill(text, width=79, break_long_words=False, initial_indent=prefix, subsequent_indent=prefix) + '\n'