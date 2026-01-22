import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_numbering(self, text):
    """ We handle the special extension for generic numbering for
            tables, figures etc.
        """
    self.regex_defns = re.compile("\n            \\[\\#(\\w+) # the counter.  Open square plus hash plus a word \\1\n            ([^@]*)   # Some optional characters, that aren't an @. \\2\n            @(\\w+)       # the id.  Should this be normed? \\3\n            ([^\\]]*)\\]   # The rest of the text up to the terminating ] \\4\n            ", re.VERBOSE)
    self.regex_subs = re.compile('\\[@(\\w+)\\s*\\]')
    counters = {}
    references = {}
    replacements = []
    definition_html = '<figcaption class="{}" id="counter-ref-{}">{}{}{}</figcaption>'
    reference_html = '<a class="{}" href="#counter-ref-{}">{}</a>'
    for match in self.regex_defns.finditer(text):
        if len(match.groups()) != 4:
            continue
        counter = match.group(1)
        text_before = match.group(2).strip()
        ref_id = match.group(3)
        text_after = match.group(4)
        number = counters.get(counter, 1)
        references[ref_id] = (number, counter)
        replacements.append((match.start(0), definition_html.format(counter, ref_id, text_before, number, text_after), match.end(0)))
        counters[counter] = number + 1
    for repl in reversed(replacements):
        text = text[:repl[0]] + repl[1] + text[repl[2]:]
    for match in reversed(list(self.regex_subs.finditer(text))):
        number, counter = references.get(match.group(1), (None, None))
        if number is not None:
            repl = reference_html.format(counter, match.group(1), number)
        else:
            repl = reference_html.format(match.group(1), 'countererror', '?' + match.group(1) + '?')
        if 'smarty-pants' in self.extras:
            repl = repl.replace('"', self._escape_table['"'])
        text = text[:match.start()] + repl + text[match.end():]
    return text