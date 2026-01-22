import io
import re
from Bio.SeqFeature import SeqFeature, SimpleLocation, Position
def _read_dr(record, value):
    cols = value.rstrip('.').split('; ')
    record.cross_references.append(tuple(cols))