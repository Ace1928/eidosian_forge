import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _encode_email_address(self, addr):
    chars = [_xml_encode_email_char_at_random(ch) for ch in 'mailto:' + addr]
    addr = '<a href="%s">%s</a>' % (''.join(chars), ''.join(chars[7:]))
    return addr