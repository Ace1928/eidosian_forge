from collections import namedtuple
import json
import logging
import pprint
import re
def _quick_help(self, nested=False):
    """:param nested: True if help is requested directly for this command
                    and False when help is requested for a list of possible
                    completions.
        """
    if nested:
        return (self.command_path(), None, self.help_msg)
    else:
        return (self.command_path(), self.param_help_msg, self.help_msg)