from __future__ import (absolute_import, division, print_function)
from ansible.plugins import AnsiblePlugin
from ansible.utils.path import basedir
from ansible.utils.display import Display
class BaseVarsPlugin(AnsiblePlugin):
    """
    Loads variables for groups and/or hosts
    """
    is_stateless = False

    def __init__(self):
        """ constructor """
        super(BaseVarsPlugin, self).__init__()
        self._display = display

    def get_vars(self, loader, path, entities):
        """ Gets variables. """
        self._basedir = basedir(path)