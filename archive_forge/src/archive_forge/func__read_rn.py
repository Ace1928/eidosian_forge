import io
import re
from Bio.SeqFeature import SeqFeature, SimpleLocation, Position
def _read_rn(reference, rn):
    words = rn.split(None, 1)
    number = words[0]
    assert number.startswith('[') and number.endswith(']'), f'Missing brackets {number}'
    reference.number = int(number[1:-1])
    if len(words) > 1:
        evidence = words[1]
        assert evidence.startswith('{') and evidence.endswith('}'), f'Missing braces {evidence}'
        reference.evidence = evidence[1:-1].split('|')