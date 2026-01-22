from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import six
def MakeArgKwargs(self):
    """Returns argparse kwargs shared between all concept types."""
    return {'help': self.BuildHelpText(), 'required': self.IsArgRequired(), 'hidden': self.hidden}