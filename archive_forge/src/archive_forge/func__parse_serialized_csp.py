import dataclasses
from typing import Collection
from werkzeug.datastructures import Headers
from werkzeug import http
from tensorboard.util import tb_logging
def _parse_serialized_csp(self, csp_text):
    csp_srcs = csp_text.split(';')
    policy = []
    for token in csp_srcs:
        token = token.strip()
        if not token:
            continue
        token_frag = token.split(None, 1)
        name = token_frag[0]
        values = token_frag[1] if len(token_frag) == 2 else ''
        name = name.lower()
        value = values.split()
        directive = Directive(name=name, value=value)
        policy.append(directive)
    return policy