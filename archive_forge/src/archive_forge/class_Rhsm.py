from __future__ import (absolute_import, division, print_function)
import os
import re
import shutil
import tempfile
import types
from ansible.module_utils.six.moves import configparser
class Rhsm(RegistrationBase):
    """
    DEPRECATION WARNING

    This class is deprecated and will be removed in community.general 9.0.0.
    There is no replacement for it; please contact the community.general
    maintainers in case you are using it.
    """

    def __init__(self, module, username=None, password=None):
        RegistrationBase.__init__(self, module, username, password)
        self.config = self._read_config()
        self.module = module
        self.module.deprecate('The Rhsm class is deprecated with no replacement.', version='9.0.0', collection_name='community.general')

    def _read_config(self, rhsm_conf='/etc/rhsm/rhsm.conf'):
        """
            Load RHSM configuration from /etc/rhsm/rhsm.conf.
            Returns:
             * ConfigParser object
        """
        cp = configparser.ConfigParser()
        cp.read(rhsm_conf)

        def get_option_default(self, key, default=''):
            sect, opt = key.split('.', 1)
            if self.has_section(sect) and self.has_option(sect, opt):
                return self.get(sect, opt)
            else:
                return default
        cp.get_option = types.MethodType(get_option_default, cp, configparser.ConfigParser)
        return cp

    def enable(self):
        """
            Enable the system to receive updates from subscription-manager.
            This involves updating affected yum plugins and removing any
            conflicting yum repositories.
        """
        RegistrationBase.enable(self)
        self.update_plugin_conf('rhnplugin', False)
        self.update_plugin_conf('subscription-manager', True)

    def configure(self, **kwargs):
        """
            Configure the system as directed for registration with RHN
            Raises:
              * Exception - if error occurs while running command
        """
        args = ['subscription-manager', 'config']
        for k, v in kwargs.items():
            if re.search('^(system|rhsm)_', k):
                args.append('--%s=%s' % (k.replace('_', '.'), v))
        self.module.run_command(args, check_rc=True)

    @property
    def is_registered(self):
        """
            Determine whether the current system
            Returns:
              * Boolean - whether the current system is currently registered to
                          RHN.
        """
        args = ['subscription-manager', 'identity']
        rc, stdout, stderr = self.module.run_command(args, check_rc=False)
        if rc == 0:
            return True
        else:
            return False

    def register(self, username, password, autosubscribe, activationkey):
        """
            Register the current system to the provided RHN server
            Raises:
              * Exception - if error occurs while running command
        """
        args = ['subscription-manager', 'register']
        if activationkey:
            args.append('--activationkey "%s"' % activationkey)
        else:
            if autosubscribe:
                args.append('--autosubscribe')
            if username:
                args.extend(['--username', username])
            if password:
                args.extend(['--password', password])
        rc, stderr, stdout = self.module.run_command(args, check_rc=True)

    def unsubscribe(self):
        """
            Unsubscribe a system from all subscribed channels
            Raises:
              * Exception - if error occurs while running command
        """
        args = ['subscription-manager', 'unsubscribe', '--all']
        rc, stderr, stdout = self.module.run_command(args, check_rc=True)

    def unregister(self):
        """
            Unregister a currently registered system
            Raises:
              * Exception - if error occurs while running command
        """
        args = ['subscription-manager', 'unregister']
        rc, stderr, stdout = self.module.run_command(args, check_rc=True)
        self.update_plugin_conf('rhnplugin', False)
        self.update_plugin_conf('subscription-manager', False)

    def subscribe(self, regexp):
        """
            Subscribe current system to available pools matching the specified
            regular expression
            Raises:
              * Exception - if error occurs while running command
        """
        available_pools = RhsmPools(self.module)
        for pool in available_pools.filter(regexp):
            pool.subscribe()