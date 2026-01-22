from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
def _apply_inflections(self, word, rules):
    result = word
    if word != '' and result.lower() not in self.inflections.uncountables:
        for rule, replacement in rules:
            result = re.sub(rule, replacement, result)
            if result != word:
                break
    return result