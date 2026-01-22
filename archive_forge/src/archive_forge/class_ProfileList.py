import os
from traitlets.config.application import Application
from IPython.core.application import (
from IPython.core.profiledir import ProfileDir
from IPython.utils.importstring import import_item
from IPython.paths import get_ipython_dir, get_ipython_package_dir
from traitlets import Unicode, Bool, Dict, observe
class ProfileList(Application):
    name = u'ipython-profile'
    description = list_help
    examples = _list_examples
    aliases = Dict({'ipython-dir': 'ProfileList.ipython_dir', 'log-level': 'Application.log_level'})
    flags = Dict(dict(debug=({'Application': {'log_level': 0}}, 'Set Application.log_level to 0, maximizing log output.')))
    ipython_dir = Unicode(get_ipython_dir(), help='\n        The name of the IPython directory. This directory is used for logging\n        configuration (through profiles), history storage, etc. The default\n        is usually $HOME/.ipython. This options can also be specified through\n        the environment variable IPYTHONDIR.\n        ').tag(config=True)

    def _print_profiles(self, profiles):
        """print list of profiles, indented."""
        for profile in profiles:
            print('    %s' % profile)

    def list_profile_dirs(self):
        profiles = list_bundled_profiles()
        if profiles:
            print()
            print('Available profiles in IPython:')
            self._print_profiles(profiles)
            print()
            print('    The first request for a bundled profile will copy it')
            print('    into your IPython directory (%s),' % self.ipython_dir)
            print('    where you can customize it.')
        profiles = list_profiles_in(self.ipython_dir)
        if profiles:
            print()
            print('Available profiles in %s:' % self.ipython_dir)
            self._print_profiles(profiles)
        profiles = list_profiles_in(os.getcwd())
        if profiles:
            print()
            print('Profiles from CWD have been removed for security reason, see CVE-2022-21699:')
        print()
        print('To use any of the above profiles, start IPython with:')
        print('    ipython --profile=<name>')
        print()

    def start(self):
        self.list_profile_dirs()