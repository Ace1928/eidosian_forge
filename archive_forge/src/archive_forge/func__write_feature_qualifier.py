import warnings
from datetime import datetime
from Bio import BiopythonWarning
from Bio import SeqFeature
from Bio import SeqIO
from Bio.GenBank.Scanner import _ImgtScanner
from Bio.GenBank.Scanner import EmblScanner
from Bio.GenBank.Scanner import GenBankScanner
from Bio.Seq import UndefinedSequenceError
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _write_feature_qualifier(self, key, value=None, quote=None):
    if value is None:
        self.handle.write(f'{self.QUALIFIER_INDENT_STR}/{key}\n')
        return
    if isinstance(value, str):
        value = value.replace('"', '""')
    if quote is None:
        if isinstance(value, int) or key in self.FTQUAL_NO_QUOTE:
            quote = False
        else:
            quote = True
    if quote:
        line = f'{self.QUALIFIER_INDENT_STR}/{key}="{value}"'
    else:
        line = f'{self.QUALIFIER_INDENT_STR}/{key}={value}'
    if len(line) <= self.MAX_WIDTH:
        self.handle.write(line + '\n')
        return
    while line.lstrip():
        if len(line) <= self.MAX_WIDTH:
            self.handle.write(line + '\n')
            return
        for index in range(min(len(line) - 1, self.MAX_WIDTH), self.QUALIFIER_INDENT + 1, -1):
            if line[index] == ' ':
                break
        if line[index] != ' ':
            index = self.MAX_WIDTH
        assert index <= self.MAX_WIDTH
        self.handle.write(line[:index] + '\n')
        line = self.QUALIFIER_INDENT_STR + line[index:].lstrip()