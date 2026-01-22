from __future__ import (absolute_import, division, print_function)
import os
import re
import shutil
import tempfile
import types
from ansible.module_utils.six.moves import configparser
class RhsmPool(object):
    """
    Convenience class for housing subscription information

    DEPRECATION WARNING

    This class is deprecated and will be removed in community.general 9.0.0.
    There is no replacement for it; please contact the community.general
    maintainers in case you are using it.
    """

    def __init__(self, module, **kwargs):
        self.module = module
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.module.deprecate('The RhsmPool class is deprecated with no replacement.', version='9.0.0', collection_name='community.general')

    def __str__(self):
        return str(self.__getattribute__('_name'))

    def subscribe(self):
        args = 'subscription-manager subscribe --pool %s' % self.PoolId
        rc, stdout, stderr = self.module.run_command(args, check_rc=True)
        if rc == 0:
            return True
        else:
            return False