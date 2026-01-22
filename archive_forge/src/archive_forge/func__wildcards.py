import random
import re
def _wildcards(self, response, match):
    pos = response.find('%')
    while pos >= 0:
        num = int(response[pos + 1:pos + 2])
        response = response[:pos] + self._substitute(match.group(num)) + response[pos + 2:]
        pos = response.find('%')
    return response