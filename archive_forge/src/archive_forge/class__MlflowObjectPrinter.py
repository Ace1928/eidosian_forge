import pprint
from abc import abstractmethod
class _MlflowObjectPrinter:

    def __init__(self):
        super().__init__()
        self.printer = pprint.PrettyPrinter()

    def to_string(self, obj):
        if isinstance(obj, _MlflowObject):
            return f'<{get_classname(obj)}: {self._entity_to_string(obj)}>'
        return self.printer.pformat(obj)

    def _entity_to_string(self, entity):
        return ', '.join([f'{key}={self.to_string(value)}' for key, value in entity])