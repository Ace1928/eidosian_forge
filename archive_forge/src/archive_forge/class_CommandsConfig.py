import os
from setuptools.command import easy_install
from pbr.hooks import base
from pbr import options
from pbr import packaging
class CommandsConfig(base.BaseConfig):
    section = 'global'

    def __init__(self, config):
        super(CommandsConfig, self).__init__(config)
        self.commands = self.config.get('commands', '')

    def save(self):
        self.config['commands'] = self.commands
        super(CommandsConfig, self).save()

    def add_command(self, command):
        self.commands = '%s\n%s' % (self.commands, command)

    def hook(self):
        self.add_command('pbr.packaging.LocalEggInfo')
        self.add_command('pbr.packaging.LocalSDist')
        self.add_command('pbr.packaging.LocalInstallScripts')
        self.add_command('pbr.packaging.LocalDevelop')
        self.add_command('pbr.packaging.LocalRPMVersion')
        self.add_command('pbr.packaging.LocalDebVersion')
        if os.name != 'nt':
            easy_install.get_script_args = packaging.override_get_script_args
        if os.path.exists('.testr.conf') and packaging.have_testr():
            self.add_command('pbr.packaging.TestrTest')
        elif self.config.get('nosetests', False) and packaging.have_nose():
            self.add_command('pbr.packaging.NoseTest')
        use_egg = options.get_boolean_option(self.pbr_config, 'use-egg', 'PBR_USE_EGG')
        if 'manpages' in self.pbr_config or not use_egg:
            self.add_command('pbr.packaging.LocalInstall')
        else:
            self.add_command('pbr.packaging.InstallWithGit')