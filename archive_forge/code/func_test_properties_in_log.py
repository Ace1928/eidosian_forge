import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_properties_in_log(self):
    """Log includes the custom properties returned by the registered
        handlers.
        """
    wt = self.make_standard_commit('test_properties_in_log')

    def trivial_custom_prop_handler(revision):
        return {'test_prop': 'test_value'}
    log.properties_handler_registry.register('trivial_custom_prop_handler', trivial_custom_prop_handler)
    self.assertFormatterResult(b'------------------------------------------------------------\nrevno: 1\ntest_prop: test_value\nauthor: John Doe <jdoe@example.com>\ncommitter: Lorem Ipsum <test@example.com>\nbranch nick: test_properties_in_log\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  add a\n', wt.branch, log.LongLogFormatter)