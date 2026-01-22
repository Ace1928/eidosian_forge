from __future__ import absolute_import, division, print_function
import json
from threading import RLock
from ansible.module_utils.six import itervalues
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
class CliProvider(ProviderBase):
    supported_connections = ('network_cli',)

    @property
    def capabilities(self):
        if not hasattr(self, '_capabilities'):
            resp = self.from_json(self.connection.get_capabilities())
            setattr(self, '_capabilities', resp)
        return getattr(self, '_capabilities')

    def get_config_context(self, config, path, indent=1):
        if config is not None:
            netcfg = NetworkConfig(indent=indent, contents=config)
            try:
                config = netcfg.get_block_config(to_list(path))
            except ValueError:
                config = None
            return config

    def render(self, config=None):
        raise NotImplementedError(self.__class__.__name__)

    def cli(self, command):
        try:
            if not hasattr(self, '_command_output'):
                setattr(self, '_command_output', {})
            return self._command_output[command]
        except KeyError:
            out = self.connection.get(command)
            try:
                out = json.loads(out)
            except ValueError:
                pass
            self._command_output[command] = out
            return out

    def get_facts(self, subset=None):
        return self.populate()

    def edit_config(self, config=None):
        commands = self.render(config)
        if commands and self.check_mode is False:
            self.connection.edit_config(commands)
        return commands