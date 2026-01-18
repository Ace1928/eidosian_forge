import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def test_search_checks_for_duplicate_prefixes(self):
    """Its an error when there are multiple indices with the same prefix."""
    indices = help.HelpIndices()
    indices.search_path = [help_topics.HelpTopicIndex(), help_topics.HelpTopicIndex()]
    self.assertRaises(errors.DuplicateHelpPrefix, indices.search, None)