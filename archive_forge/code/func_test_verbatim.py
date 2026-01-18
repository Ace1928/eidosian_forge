import logging
import os
from inspect import currentframe, getframeinfo
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import (
def test_verbatim(self):
    self.handler.setFormatter(LegacyPyomoFormatter(base=os.path.dirname(__file__), verbosity=lambda: logger.isEnabledFor(logging.DEBUG)))
    msg = 'This is a long message\n\n   ```\nWith some \ninternal\nverbatim \n  - including a\n    long list\n  - and a short list \n  ```\n\nAnd some \ninternal\nnon-verbatim \n  - including a\n    long list\n  - and a short list \n\nAnd a section\n~~~~~~~~~~~~~\n\n  | and\n  | a line\n  | block\n\nAnd a\nquoted literal::\n\n>> he said\n>\n> and they replied\n\nthis is\noutside the quote\n\nindented literal::\n\n    Here is\n       an indented\n\n    literal\n    with a blank line\n\nFinally, an invalid::\n\nquote\nblock\n'
    logger.setLevel(logging.WARNING)
    logger.warning(msg)
    ans = 'WARNING: This is a long message\n\n    With some \n    internal\n    verbatim \n      - including a\n        long list\n      - and a short list \n\n    And some internal non-verbatim\n      - including a long list\n      - and a short list\n\n    And a section\n    ~~~~~~~~~~~~~\n\n      | and\n      | a line\n      | block\n\n    And a quoted literal::\n\n    >> he said\n    >\n    > and they replied\n\n    this is outside the quote\n\n    indented literal::\n\n        Here is\n           an indented\n\n        literal\n        with a blank line\n\n    Finally, an invalid::\n\n    quote block\n'
    self.maxDiff = None
    self.assertEqual(self.stream.getvalue(), ans)