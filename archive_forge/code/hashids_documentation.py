from __future__ import absolute_import, division, print_function
from ansible.errors import (
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.collections import is_sequence
Decodes a YouTube-like hash to a sequence of ints

       :hashid: Hash string to decode
       :salt: String to use as salt when hashing
       :alphabet: String of 16 or more unique characters to produce a hash
       :min_length: Minimum length of hash produced
    