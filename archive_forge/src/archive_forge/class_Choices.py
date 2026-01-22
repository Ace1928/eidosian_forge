import enum
import warnings
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import Promise
from django.utils.version import PY311, PY312
class Choices(enum.Enum, metaclass=ChoicesType):
    """Class for creating enumerated choices."""
    if PY311:
        do_not_call_in_templates = enum.nonmember(True)
    else:

        @property
        def do_not_call_in_templates(self):
            return True

    @enum_property
    def label(self):
        return self._label_

    def __repr__(self):
        return f'{self.__class__.__qualname__}.{self._name_}'