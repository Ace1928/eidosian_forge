import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def test_search_returns_index_and_results(self):
    """Searching should return help topics with their index"""

    class CannedIndex:

        def __init__(self, prefix, search_result):
            self.prefix = prefix
            self.result = search_result

        def get_topics(self, topic):
            return self.result
    index = help.HelpIndices()
    index_one = CannedIndex('1', ['a'])
    index_two = CannedIndex('2', ['b', 'c'])
    index.search_path = [index_one, index_two]
    self.assertEqual([(index_one, 'a'), (index_two, 'b'), (index_two, 'c')], index.search(None))