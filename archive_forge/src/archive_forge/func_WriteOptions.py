from __future__ import absolute_import
import os
def WriteOptions(self):
    """Writes the 'options' field based on other settings."""
    options = []
    if self.public:
        options.append('public')
    if self.dynamic:
        options.append('dynamic')
    if self.failfast:
        options.append('failfast')
    if options:
        self.options = ', '.join(options)
    else:
        self.options = None
    return self