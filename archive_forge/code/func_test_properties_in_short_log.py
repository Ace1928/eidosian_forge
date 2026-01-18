import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_properties_in_short_log(self):
    """Log includes the custom properties returned by the registered
        handlers.
        """
    wt = self.make_standard_commit('test_properties_in_short_log')

    def trivial_custom_prop_handler(revision):
        return {'test_prop': 'test_value'}
    log.properties_handler_registry.register('trivial_custom_prop_handler', trivial_custom_prop_handler)
    self.assertFormatterResult(b'    1 John Doe\t2005-11-22\n      test_prop: test_value\n      add a\n\n', wt.branch, log.ShortLogFormatter)