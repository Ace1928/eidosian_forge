from inspect import isclass
from .props import props
class SubclassWithMeta_Meta(type):
    _meta = None

    def __str__(cls):
        if cls._meta:
            return cls._meta.name
        return cls.__name__

    def __repr__(cls):
        return f'<{cls.__name__} meta={repr(cls._meta)}>'