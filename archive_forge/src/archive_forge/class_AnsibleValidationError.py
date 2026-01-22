from __future__ import absolute_import, division, print_function
class AnsibleValidationError(Exception):
    """Single argument spec validation error"""

    def __init__(self, message):
        super(AnsibleValidationError, self).__init__(message)
        self.error_message = message
        'The error message passed in when the exception was raised.'

    @property
    def msg(self):
        """The error message passed in when the exception was raised."""
        return self.args[0]