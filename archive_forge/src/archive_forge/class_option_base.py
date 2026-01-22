from distutils.util import convert_path
from distutils import log
from distutils.errors import DistutilsOptionError
import distutils
import os
import configparser
from setuptools import Command
class option_base(Command):
    """Abstract base class for commands that mess with config files"""
    user_options = [('global-config', 'g', 'save options to the site-wide distutils.cfg file'), ('user-config', 'u', "save options to the current user's pydistutils.cfg file"), ('filename=', 'f', 'configuration file to use (default=setup.cfg)')]
    boolean_options = ['global-config', 'user-config']

    def initialize_options(self):
        self.global_config = None
        self.user_config = None
        self.filename = None

    def finalize_options(self):
        filenames = []
        if self.global_config:
            filenames.append(config_file('global'))
        if self.user_config:
            filenames.append(config_file('user'))
        if self.filename is not None:
            filenames.append(self.filename)
        if not filenames:
            filenames.append(config_file('local'))
        if len(filenames) > 1:
            raise DistutilsOptionError('Must specify only one configuration file option', filenames)
        self.filename, = filenames