import re
import types
from pyomo.common.sorting import sorted_robust
def wrap_reStructuredText(docstr, wrapper):
    """A text wrapper that honors paragraphs and basic reStructuredText markup

    This wraps `textwrap.fill()` to first separate the incoming text by
    paragraphs before using ``wrapper`` to wrap each one.  It includes a
    basic (partial) parser for reStructuredText format to attempt to
    avoid wrapping structural elements like section headings, bullet /
    enumerated lists, and tables.

    Parameters
    ----------
    docstr : str
        The incoming string to parse and wrap

    wrapper : `textwrap.TextWrap`
        The configured `TextWrap` object to use for wrapping paragraphs.
        While the object will be reconfigured within this function, it
        will be restored to its original state upon exit.

    """
    paragraphs = [(None, None, None)]
    literal_block = False
    verbatim = False
    for line in docstr.rstrip().splitlines():
        leading = _indentation_re.match(line).group()
        content = line.strip()
        if not content:
            if literal_block:
                if literal_block[0] == 2:
                    literal_block = False
            elif paragraphs[-1][2] and ''.join(paragraphs[-1][2]).endswith('::'):
                literal_block = (0, paragraphs[-1][1])
            paragraphs.append((None, None, None))
            continue
        if literal_block:
            if literal_block[0] == 0:
                if len(literal_block[1]) < len(leading):
                    literal_block = (1, leading)
                    paragraphs.append((None, None, line))
                    continue
                elif len(literal_block[1]) == len(leading) and content[0] in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
                    literal_block = (2, leading)
                    paragraphs.append((None, None, line))
                    continue
                else:
                    literal_block = False
            elif leading.startswith(literal_block[1]):
                paragraphs.append((None, None, line))
                continue
            else:
                literal_block = False
        if content == '```':
            verbatim ^= True
        elif verbatim:
            paragraphs.append((None, None, line))
        elif _verbatim_line_start.match(content):
            paragraphs.append((None, None, line))
        elif _verbatim_line.match(content):
            paragraphs.append((None, None, line))
        else:
            matchBullet = _bullet_re.match(content)
            if matchBullet:
                hang = matchBullet.group()
                paragraphs.append((leading, leading + ' ' * len(hang), [content]))
            elif paragraphs[-1][1] == leading:
                paragraphs[-1][2].append(content)
            else:
                paragraphs.append((leading, leading, [content]))
    while paragraphs and paragraphs[0][2] is None:
        paragraphs.pop(0)
    wrapper_init = (wrapper.initial_indent, wrapper.subsequent_indent)
    try:
        for i, (indent, subseq, par) in enumerate(paragraphs):
            base_indent = wrapper_init[1] if i else wrapper_init[0]
            if indent is None:
                if par is None:
                    paragraphs[i] = ''
                else:
                    paragraphs[i] = base_indent + par
                continue
            wrapper.initial_indent = base_indent + indent
            wrapper.subsequent_indent = base_indent + subseq
            paragraphs[i] = wrapper.fill(' '.join(par))
    finally:
        wrapper.initial_indent, wrapper.subsequent_indent = wrapper_init
    return '\n'.join(paragraphs)