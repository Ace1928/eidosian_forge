from yowsup.config.v1.config import Config
from yowsup.config.transforms.dict_keyval import DictKeyValTransform
from yowsup.config.transforms.dict_json import DictJsonTransform
from yowsup.config.v1.serialize import ConfigSerialize
from yowsup.common.tools import StorageTools
import logging
import os
def _load_path(self, path):
    """
        :param path:
        :type path:
        :return:
        :rtype:
        """
    logger.debug('_load_path(path=%s)' % path)
    if os.path.isfile(path):
        configtype = self.guess_type(path)
        logger.debug('Detected config type: %s' % self._type_to_str(configtype))
        if configtype in self.TYPES:
            logger.debug('Opening config for reading')
            with open(path, 'r') as f:
                data = f.read()
            datadict = self.TYPES[configtype]().reverse(data)
            return self.load_data(datadict)
        else:
            raise ValueError('Unsupported config type')
    else:
        logger.debug("_load_path couldn't find the path: %s" % path)