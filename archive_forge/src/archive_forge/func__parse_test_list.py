import os
from breezy import tests
from breezy.tests import features
from breezy.transport import memory
def _parse_test_list(self, lines, newlines_in_header=0):
    """Parse a list of lines into a tuple of 3 lists (header,body,footer)."""
    in_header = newlines_in_header != 0
    in_footer = False
    header = []
    body = []
    footer = []
    header_newlines_found = 0
    for line in lines:
        if in_header:
            if line == '':
                header_newlines_found += 1
                if header_newlines_found >= newlines_in_header:
                    in_header = False
                    continue
            header.append(line)
        elif not in_footer:
            if line.startswith('-------'):
                in_footer = True
            else:
                body.append(line)
        else:
            footer.append(line)
    if len(body) > 0 and body[-1] == '':
        body.pop()
    return (header, body, footer)