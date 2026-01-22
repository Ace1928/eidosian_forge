import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class TestLogDefaults(TestCaseForLogFormatter):

    def test_default_log_level(self):
        """
        Test to ensure that specifying 'levels=1' to make_log_request_dict
        doesn't get overwritten when using a LogFormatter that supports more
        detail.
        Fixes bug #747958.
        """
        wt = self._prepare_tree_with_merges()
        b = wt.branch

        class CustomLogFormatter(log.LogFormatter):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.revisions = []

            def get_levels(self):
                return 0

            def log_revision(self, revision):
                self.revisions.append(revision)
        log_formatter = LogCatcher()
        request = log.make_log_request_dict(limit=10)
        log.Logger(b, request).show(log_formatter)
        self.assertEqual(len(log_formatter.revisions), 3)
        del log_formatter
        log_formatter = LogCatcher()
        request = log.make_log_request_dict(limit=10, levels=1)
        log.Logger(b, request).show(log_formatter)
        self.assertEqual(len(log_formatter.revisions), 2)