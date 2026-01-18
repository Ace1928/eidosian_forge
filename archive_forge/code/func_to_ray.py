from modin.config import Engine, IsExperimental, StorageFormat
from modin.core.execution.dispatching.factories import factories
from modin.utils import _inherit_docstrings, get_current_execution
@classmethod
@_inherit_docstrings(factories.BaseFactory._to_ray)
def to_ray(cls, modin_obj):
    return cls.get_factory()._to_ray(modin_obj)