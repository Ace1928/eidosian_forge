from collections import Counter
from textwrap import dedent
from kombu.utils.encoding import bytes_to_str, safe_str
def _enc(self, s):
    return s.encode('utf-8', 'ignore')