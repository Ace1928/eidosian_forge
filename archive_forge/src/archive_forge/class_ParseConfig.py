from __future__ import unicode_literals
import logging
import re
from cmakelang.parse import util as parse_util
from cmakelang.parse.funs import standard_funs
from cmakelang import markup
from cmakelang.config_util import (
class ParseConfig(ConfigObject):
    """Options affecting listfile parsing"""
    _field_registry = []
    additional_commands = FieldDescriptor(ADDITIONAL_COMMANDS_DEMO, 'Specify structure for custom cmake functions')
    override_spec = FieldDescriptor({}, 'Override configurations per-command where available')
    vartags = FieldDescriptor([], 'Specify variable tags.')
    proptags = FieldDescriptor([], 'Specify property tags.')

    def __init__(self, **kwargs):
        self.fn_spec = parse_util.CommandSpec('<root>')
        self.vartags_ = []
        self.proptags_ = []
        super(ParseConfig, self).__init__(**kwargs)

    def _update_derived(self):
        if self.additional_commands is not None:
            for command_name, spec in self.additional_commands.items():
                self.fn_spec.add(command_name, **spec)
        for pathkeystr, value in self.override_spec.items():
            parse_util.apply_overrides(self.fn_spec, pathkeystr, value)
        self.vartags_ = [(re.compile(pattern, re.IGNORECASE), tags) for pattern, tags in BUILTIN_VARTAGS + self.vartags]
        self.proptags_ = [(re.compile(pattern, re.IGNORECASE), tags) for pattern, tags in BUILTIN_PROPTAGS + self.proptags]