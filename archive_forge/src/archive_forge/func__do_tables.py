import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_tables(self, text):
    """Copying PHP-Markdown and GFM table syntax. Some regex borrowed from
        https://github.com/michelf/php-markdown/blob/lib/Michelf/Markdown.php#L2538
        """
    less_than_tab = self.tab_width - 1
    table_re = re.compile('\n                (?:(?<=\\n)|\\A\\n?)             # leading blank line\n\n                ^[ ]{0,%d}                      # allowed whitespace\n                (.*[|].*)[ ]*\\n                   # $1: header row (at least one pipe)\n\n                ^[ ]{0,%d}                      # allowed whitespace\n                (                               # $2: underline row\n                    # underline row with leading bar\n                    (?:  \\|\\ *:?-+:?\\ *  )+  \\|? \\s?[ ]*\\n\n                    |\n                    # or, underline row without leading bar\n                    (?:  \\ *:?-+:?\\ *\\|  )+  (?:  \\ *:?-+:?\\ *  )? \\s?[ ]*\\n\n                )\n\n                (                               # $3: data rows\n                    (?:\n                        ^[ ]{0,%d}(?!\\ )         # ensure line begins with 0 to less_than_tab spaces\n                        .*\\|.*[ ]*\\n\n                    )+\n                )\n            ' % (less_than_tab, less_than_tab, less_than_tab), re.M | re.X)
    return table_re.sub(self._table_sub, text)