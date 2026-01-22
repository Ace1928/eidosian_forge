from __future__ import print_function
import os
from .. import CatBoostError
from ..eval.log_config import get_eval_logger
from .utils import make_dirs_if_not_exists
class FoldStorage(object):
    """
    Base class.
    """
    default_dir = 'folds'

    @staticmethod
    def remove_dir():
        """
        Remove default directory for folds if there're no files nut models. In other way it raises warning.

        Args:
            :return: Nothing.

        """
        try:
            if os.path.exists(_FoldFile.default_dir):
                os.rmdir(_FoldFile.default_dir)
        except OSError as err:
            get_eval_logger().warning(err.message)

    def __init__(self, fold, storage_name, sep, column_description):
        self._fold = fold
        self._storage_name = storage_name
        self._column_description = column_description
        self._sep = sep
        self._size = 0

    def get_separator(self):
        """
        Args:
            :return: (str) Delimiter for data used when we saved fold to file.

        """
        return self._sep

    def __str__(self):
        return self._storage_name

    def column_description(self):
        """
        Args:
            :return: (str) Path to the column description.

        """
        return self._column_description

    def contains_group_id(self, group_id):
        """
        Args:
            :param group_id: (int) The number of group we want to check.
            :return: True if fold contains line or lines with that group id.

        """
        return group_id in self._fold

    def open(self):
        raise NotImplementedError("The base class don't have delete method. Please, use successor.")

    def close(self):
        raise NotImplementedError("The base class don't have delete method. Please, use successor.")

    def delete(self):
        raise NotImplementedError("The base class don't have delete method. Please, use successor.")