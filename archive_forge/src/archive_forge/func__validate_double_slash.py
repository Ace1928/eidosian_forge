import copy
import re
import types
from .ucre import build_re
def _validate_double_slash(self, text, pos):
    tail = text[pos:]
    if not self.re.get('not_http'):
        self.re['not_http'] = '^' + self.re['src_auth'] + '(?:localhost|(?:(?:' + self.re['src_domain'] + ')\\.)+' + self.re['src_domain_root'] + ')' + self.re['src_port'] + self.re['src_host_terminator'] + self.re['src_path']
    founds = re.search(self.re['not_http'], tail, flags=re.IGNORECASE)
    if founds:
        if pos >= 3 and text[pos - 3] == ':':
            return 0
        if pos >= 3 and text[pos - 3] == '/':
            return 0
        return len(founds.group(0))
    return 0