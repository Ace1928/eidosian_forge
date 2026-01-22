from __future__ import unicode_literals
import logging
import re
from cmakelang.parse import util as parse_util
from cmakelang.parse.funs import standard_funs
from cmakelang import markup
from cmakelang.config_util import (
class MiscConfig(ConfigObject):
    """Miscellaneous configurations options."""
    _field_registry = []
    per_command = FieldDescriptor({}, 'A dictionary containing any per-command configuration overrides. Currently only `command_case` is supported.')

    def _update_derived(self):
        self.per_command_ = standard_funs.get_default_config()
        for command, cdict in self.per_command.items():
            if not isinstance(cdict, dict):
                logging.warning('Invalid override of type %s for %s', type(cdict), command)
                continue
            command = command.lower()
            if command not in self.per_command_:
                self.per_command_[command] = {}
            self.per_command_[command].update(cdict)

    def __init__(self, **kwargs):
        self.per_command_ = {}
        super(MiscConfig, self).__init__(**kwargs)