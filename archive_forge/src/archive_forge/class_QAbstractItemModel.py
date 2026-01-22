import enum
import importlib
import_errors = []
class QAbstractItemModel(object):
    """
        A dummy QAbstractItemModel class to allow some testing without PyQt
        """

    def __init__(*args, **kwargs):
        pass