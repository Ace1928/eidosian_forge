import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
class TestConfigOptionIndex(TestHelp):
    """Tests for the HelpCommandIndex class."""

    def setUp(self):
        super().setUp()
        self.index = help_topics.ConfigOptionHelpIndex()

    def test_get_topics_None(self):
        """Searching for None returns an empty list."""
        self.assertEqual([], self.index.get_topics(None))

    def test_get_topics_no_topic(self):
        self.assertEqual([], self.index.get_topics('nothing by this name'))

    def test_prefix(self):
        self.assertEqual('configuration/', self.index.prefix)

    def test_get_topic_with_prefix(self):
        topics = self.index.get_topics('configuration/default_format')
        self.assertLength(1, topics)
        opt = topics[0]
        self.assertIsInstance(opt, config.Option)
        self.assertEqual('default_format', opt.name)