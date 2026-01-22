import re
from collections import OrderedDict
from typing import Any, Optional
def _group_challenges(tokens: list) -> list:
    challenges = []
    while tokens:
        j = 1
        if len(tokens) == 1:
            pass
        elif tokens[1][0] == 'comma':
            pass
        elif tokens[1][0] == 'token':
            j = 2
        else:
            while j < len(tokens) and tokens[j][0] == 'pair':
                j += 2
            j -= 1
        challenges.append((tokens[0][1], tokens[1:j]))
        tokens[:j + 1] = []
    return challenges