from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
class TransactionFile(object):
    """Context for reading/writing from/to a transaction file."""

    def __init__(self, trans_file_path, mode='r'):
        if not os.path.isfile(trans_file_path):
            raise TransactionFileNotFound('Transaction not found at [{0}]'.format(trans_file_path))
        self.__trans_file_path = trans_file_path
        try:
            if mode == 'r':
                self.__trans_file = files.FileReader(trans_file_path)
            elif mode == 'w':
                self.__trans_file = files.FileWriter(trans_file_path)
            else:
                raise ValueError('Unrecognized mode [{}]'.format(mode))
        except IOError as exp:
            msg = 'Unable to open transaction [{0}] because [{1}]'
            msg = msg.format(trans_file_path, exp)
            raise UnableToAccessTransactionFile(msg)

    def __enter__(self):
        return self.__trans_file

    def __exit__(self, typ, value, traceback):
        self.__trans_file.close()
        if typ is IOError or typ is yaml.Error:
            msg = 'Unable to read/write transaction [{0}] because [{1}]'
            msg = msg.format(self.__trans_file_path, value)
            raise UnableToAccessTransactionFile(msg)