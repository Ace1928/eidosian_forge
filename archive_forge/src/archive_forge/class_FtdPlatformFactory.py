from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import urlparse
class FtdPlatformFactory(object):

    @staticmethod
    def create(model, module_params):
        for cls in AbstractFtdPlatform.__subclasses__():
            if cls.supports_ftd_model(model):
                return cls(module_params)
        raise ValueError("FTD model '%s' is not supported by this module." % model)