import configparser as config_parser
import os
from tempest.lib.cli import base
def zun(self, action, flags='', params='', parse=True):
    """Return parsed list of dicts with basic item info.

        :param action: the cli command to run using Container
        :type action: string
        :param flags: any optional cli flags to use
        :type flags: string
        :param params: any optional positional args to use
        :type params: string
        :param parse: return parsed list or raw output
        :type parse: bool
        """
    output = self._zun(action=action, flags=flags, params=params)
    return self.parser.listing(output) if parse else output