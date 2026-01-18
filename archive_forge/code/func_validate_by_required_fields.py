from __future__ import absolute_import, division, print_function
import os
def validate_by_required_fields(self, *field_names):
    missing = [field for field in field_names if self._options.get_option_default(field) is None]
    if missing:
        raise HashiVaultValueError('Authentication method %s requires options %r to be set, but these are missing: %r' % (self.NAME, field_names, missing))