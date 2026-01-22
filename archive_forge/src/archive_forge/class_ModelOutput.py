import inspect
import tempfile
from collections import OrderedDict, UserDict
from collections.abc import MutableMapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial
from typing import Any, ContextManager, Iterable, List, Tuple
import numpy as np
from packaging import version
from .import_utils import get_torch_version, is_flax_available, is_tf_available, is_torch_available, is_torch_fx_proxy
class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __init_subclass__(cls) -> None:
        """Register subclasses as pytree nodes.

        This is necessary to synchronize gradients when using `torch.nn.parallel.DistributedDataParallel` with
        `static_graph=True` with modules that output `ModelOutput` subclasses.
        """
        if is_torch_available():
            if version.parse(get_torch_version()) >= version.parse('2.2'):
                _torch_pytree.register_pytree_node(cls, _model_output_flatten, partial(_model_output_unflatten, output_type=cls), serialized_type_name=f'{cls.__module__}.{cls.__name__}')
            else:
                _torch_pytree._register_pytree_node(cls, _model_output_flatten, partial(_model_output_unflatten, output_type=cls))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        is_modeloutput_subclass = self.__class__ != ModelOutput
        if is_modeloutput_subclass and (not is_dataclass(self)):
            raise TypeError(f'{self.__module__}.{self.__class__.__name__} is not a dataclasss. This is a subclass of ModelOutput and so must use the @dataclass decorator.')

    def __post_init__(self):
        """Check the ModelOutput dataclass.

        Only occurs if @dataclass decorator has been used.
        """
        class_fields = fields(self)
        if not len(class_fields):
            raise ValueError(f'{self.__class__.__name__} has no fields.')
        if not all((field.default is None for field in class_fields[1:])):
            raise ValueError(f'{self.__class__.__name__} should not have more than one required field.')
        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all((getattr(self, field.name) is None for field in class_fields[1:]))
        if other_fields_are_none and (not is_tensor(first_field)):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if not isinstance(element, (list, tuple)) or not len(element) == 2 or (not isinstance(element[0], str)):
                        if idx == 0:
                            self[class_fields[0].name] = first_field
                        else:
                            raise ValueError(f'Cannot set key/value for {element}. It needs to be a tuple (key, value).')
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f'You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.')

    def setdefault(self, *args, **kwargs):
        raise Exception(f'You cannot use ``setdefault`` on a {self.__class__.__name__} instance.')

    def pop(self, *args, **kwargs):
        raise Exception(f'You cannot use ``pop`` on a {self.__class__.__name__} instance.')

    def update(self, *args, **kwargs):
        raise Exception(f'You cannot use ``update`` on a {self.__class__.__name__} instance.')

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple((getattr(self, field.name) for field in fields(self)))
        return (callable, args, *remaining)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple((self[k] for k in self.keys()))