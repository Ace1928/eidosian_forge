from collections import namedtuple
import json
import logging
import pprint
import re
def _instantiate_subcommand(self, key):
    return self.subcommands[key](api=self.api, parent=self, help_formatter=self.help_formatter, resp_formatter_name=self.resp_formatter_name)