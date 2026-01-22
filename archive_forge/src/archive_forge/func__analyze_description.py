from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import text_renderer
import six
def _analyze_description(self, heading, section):
    has_errors = (self.check_for_personal_pronouns(heading, section), self.check_for_unmatched_double_backticks(heading, section), self.check_for_articles(heading, section))
    if not any(has_errors):
        self._add_no_errors_summary(heading)