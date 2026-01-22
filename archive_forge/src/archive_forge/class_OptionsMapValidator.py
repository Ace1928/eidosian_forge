import datetime
import uuid
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
class OptionsMapValidator(object):
    """Option that are passed as key(alternative) value(actual) pairs are validated on the args."""

    def __init__(self, options):
        self.key_len = max((len(option) for option in options.keys()))
        self.options = options

    def IsValid(self, s):
        if not s:
            return False
        return s[:self.key_len].upper() in self.options.keys()

    def Parse(self, s):
        if not self.IsValid(s):
            raise arg_parsers.ArgumentTypeError('Failed to parse the arg ({}). Value should be one of {}'.format(s, ', '.join(self.options.keys())))
        return self.options.get(s[:self.key_len].upper(), 'UNKNOWN')