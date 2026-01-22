import numpy as np
from keras.src.api_export import keras_export
from keras.src.backend import config
from keras.src.backend.common import global_state
from keras.src.backend.common.name_scope import current_path
from keras.src.backend.common.stateless_scope import get_stateless_scope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.naming import auto_name
class KerasVariable:

    def __init__(self, initializer, shape=None, dtype=None, trainable=True, name=None):
        name = name or auto_name(self.__class__.__name__)
        if not isinstance(name, str) or '/' in name:
            raise ValueError(f'Argument `name` must be a string and cannot contain character `/`. Received: name={name}')
        self.name = name
        parent_path = current_path()
        if parent_path:
            self.path = current_path() + '/' + self.name
        else:
            self.path = self.name
        dtype = standardize_dtype(dtype)
        self._dtype = dtype
        self._shape = None
        self._initializer = None
        self._trainable = trainable
        if isinstance(initializer, str):
            from keras.src import initializers
            initializer = initializers.get(initializer)
        if callable(initializer):
            if shape is None:
                raise ValueError(f'When creating a Variable from an initializer, the `shape` argument should be specified. Received: initializer={initializer} and shape={shape}')
        if in_stateless_scope():
            if callable(initializer):
                self._value = None
                self._initializer = initializer
                self._shape = self._validate_shape(shape)
                register_uninitialized_variable(self)
            else:
                raise ValueError('You are attempting to create a variable while in a stateless scope. This is disallowed. Make sure that all variables are created before you start using your layer/model objects.\n\nIn some cases, you might be seeing this error because you need to implement a `def build(self, input_shape)` method on your layer/model, which will create its variables.\n\nIn some other cases, you might be seeing this error because you are instantiating a `Variable` and assigning it to a layer without going through self.add_variable()/self.add_weight(). Always prefer using these methods (with a `shape` and `initializer` argument).')
        else:
            if callable(initializer):
                shape = self._validate_shape(shape)
                value = initializer(shape, dtype=dtype)
            else:
                value = initializer
            self._initialize(value)
            self._shape = tuple(self._value.shape)
        self._ndim = len(self._shape)

    def _deferred_initialize(self):
        if self._value is not None:
            raise ValueError(f'Variable {self.path} is already initialized.')
        if in_stateless_scope():
            raise ValueError('You are attempting to initialize a variable while in a stateless scope. This is disallowed. Make sure that all variables are initialized before you start using your layer/model objects.')
        value = self._initializer(self._shape, dtype=self._dtype)
        self._initialize(value)

    def _validate_shape(self, shape):
        shape = standardize_shape(shape)
        if None in shape:
            raise ValueError(f"Shapes used to initialize variables must be fully-defined (no `None` dimensions). Received: shape={shape} for variable path='{self.path}'")
        return shape

    def _maybe_autocast(self, value):
        autocast_scope = get_autocast_scope()
        if autocast_scope is not None:
            return autocast_scope.maybe_cast(value)
        return value

    def numpy(self):
        return np.array(self)

    @property
    def value(self):
        if in_stateless_scope():
            scope = get_stateless_scope()
            value = scope.get_current_value(self)
            if value is not None:
                return self._maybe_autocast(value)
        if self._value is None:
            return self._maybe_autocast(self._initializer(self._shape, dtype=self._dtype))
        return self._maybe_autocast(self._value)

    def assign(self, value):
        value = self._convert_to_tensor(value, dtype=self.dtype)
        if not shape_equal(value.shape, self.shape):
            raise ValueError(f'The shape of the target variable and the shape of the target value in `variable.assign(value)` must match. variable.shape={self.value.shape}, Received: value.shape={value.shape}. Target variable: {self}')
        if in_stateless_scope():
            scope = get_stateless_scope()
            scope.add_update((self, value))
        else:
            self._direct_assign(value)

    def assign_add(self, value):
        self.assign(self + value)

    def assign_sub(self, value):
        self.assign(self - value)

    @property
    def dtype(self):
        autocast_scope = get_autocast_scope()
        if autocast_scope is not None and is_float_dtype(self._dtype):
            return autocast_scope.dtype
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._ndim

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value

    def __repr__(self):
        return f'<KerasVariable shape={self.shape}, dtype={self.dtype}, path={self.path}>'

    def _initialize(self, value):
        raise NotImplementedError

    def _convert_to_tensor(self, value, dtype=None):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.value.__getitem__(idx)

    def __array__(self, dtype=None):
        return np.asarray(self.value.__array__(dtype))

    def __bool__(self):
        raise TypeError('A Keras Variable cannot be used as a boolean.')

    def __neg__(self):
        return self.value.__neg__()

    def __pos__(self):
        return self.value

    def __abs__(self):
        return self.value.__abs__()

    def __invert__(self):
        return self.value.__invert__()

    def __eq__(self, other):
        value = self.value
        return value.__eq__(self._convert_to_tensor(other, dtype=value.dtype))

    def __ne__(self, other):
        value = self.value
        return value.__ne__(self._convert_to_tensor(other, dtype=value.dtype))

    def __lt__(self, other):
        value = self.value
        return value.__lt__(self._convert_to_tensor(other, dtype=value.dtype))

    def __le__(self, other):
        value = self.value
        return value.__le__(self._convert_to_tensor(other, dtype=value.dtype))

    def __gt__(self, other):
        value = self.value
        return value.__gt__(self._convert_to_tensor(other, dtype=value.dtype))

    def __ge__(self, other):
        value = self.value
        return value.__ge__(self._convert_to_tensor(other, dtype=value.dtype))

    def __add__(self, other):
        value = self.value
        return value.__add__(self._convert_to_tensor(other, dtype=value.dtype))

    def __radd__(self, other):
        value = self.value
        return value.__radd__(self._convert_to_tensor(other, dtype=value.dtype))

    def __sub__(self, other):
        value = self.value
        return value.__sub__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rsub__(self, other):
        value = self.value
        return value.__rsub__(self._convert_to_tensor(other, dtype=value.dtype))

    def __mul__(self, other):
        value = self.value
        return value.__mul__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rmul__(self, other):
        value = self.value
        return value.__rmul__(self._convert_to_tensor(other, dtype=value.dtype))

    def __truediv__(self, other):
        value = self.value
        return value.__truediv__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rtruediv__(self, other):
        value = self.value
        return value.__rtruediv__(self._convert_to_tensor(other, dtype=value.dtype))

    def __floordiv__(self, other):
        value = self.value
        return value.__floordiv__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rfloordiv__(self, other):
        value = self.value
        return value.__rfloordiv__(self._convert_to_tensor(other, dtype=value.dtype))

    def __mod__(self, other):
        value = self.value
        return value.__mod__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rmod__(self, other):
        value = self.value
        return value.__rmod__(self._convert_to_tensor(other, dtype=value.dtype))

    def __pow__(self, other):
        value = self.value
        return value.__pow__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rpow__(self, other):
        value = self.value
        return value.__rpow__(self._convert_to_tensor(other, dtype=value.dtype))

    def __matmul__(self, other):
        value = self.value
        return value.__matmul__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rmatmul__(self, other):
        value = self.value
        return value.__rmatmul__(self._convert_to_tensor(other, dtype=value.dtype))

    def __and__(self, other):
        value = self.value
        return value.__and__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rand__(self, other):
        value = self.value
        return value.__rand__(self._convert_to_tensor(other, dtype=value.dtype))

    def __or__(self, other):
        value = self.value
        return value.__or__(self._convert_to_tensor(other, dtype=value.dtype))

    def __ror__(self, other):
        value = self.value
        return value.__ror__(self._convert_to_tensor(other, dtype=value.dtype))

    def __xor__(self, other):
        value = self.value
        return value.__xor__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rxor__(self, other):
        value = self.value
        return value.__rxor__(self._convert_to_tensor(other, dtype=value.dtype))