from distutils.util import convert_path
from distutils import log
from distutils.errors import DistutilsOptionError
import distutils
import os
import configparser
from setuptools import Command
class setopt(option_base):
    """Save command-line options to a file"""
    description = 'set an option in setup.cfg or another config file'
    user_options = [('command=', 'c', 'command to set an option for'), ('option=', 'o', 'option to set'), ('set-value=', 's', 'value of the option'), ('remove', 'r', 'remove (unset) the value')] + option_base.user_options
    boolean_options = option_base.boolean_options + ['remove']

    def initialize_options(self):
        option_base.initialize_options(self)
        self.command = None
        self.option = None
        self.set_value = None
        self.remove = None

    def finalize_options(self):
        option_base.finalize_options(self)
        if self.command is None or self.option is None:
            raise DistutilsOptionError('Must specify --command *and* --option')
        if self.set_value is None and (not self.remove):
            raise DistutilsOptionError('Must specify --set-value or --remove')

    def run(self):
        edit_config(self.filename, {self.command: {self.option.replace('-', '_'): self.set_value}}, self.dry_run)