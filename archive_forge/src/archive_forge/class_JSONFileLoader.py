import logging
import os
from botocore import BOTOCORE_ROOT
from botocore.compat import HAS_GZIP, OrderedDict, json
from botocore.exceptions import DataNotFoundError, UnknownServiceError
from botocore.utils import deep_merge
class JSONFileLoader:
    """Loader JSON files.

    This class can load the default format of models, which is a JSON file.

    """

    def exists(self, file_path):
        """Checks if the file exists.

        :type file_path: str
        :param file_path: The full path to the file to load without
            the '.json' extension.

        :return: True if file path exists, False otherwise.

        """
        for ext in _JSON_OPEN_METHODS:
            if os.path.isfile(file_path + ext):
                return True
        return False

    def _load_file(self, full_path, open_method):
        if not os.path.isfile(full_path):
            return
        with open_method(full_path, 'rb') as fp:
            payload = fp.read().decode('utf-8')
        logger.debug('Loading JSON file: %s', full_path)
        return json.loads(payload, object_pairs_hook=OrderedDict)

    def load_file(self, file_path):
        """Attempt to load the file path.

        :type file_path: str
        :param file_path: The full path to the file to load without
            the '.json' extension.

        :return: The loaded data if it exists, otherwise None.

        """
        for ext, open_method in _JSON_OPEN_METHODS.items():
            data = self._load_file(file_path + ext, open_method)
            if data is not None:
                return data
        return None