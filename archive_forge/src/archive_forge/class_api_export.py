from collections.abc import Sequence
import functools
import sys
from typing import Any, NamedTuple, Optional, Protocol, TypeVar
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
class api_export(object):
    """Provides ways to export symbols to the TensorFlow API."""
    _names: Sequence[str]
    _names_v1: Sequence[str]
    _api_name: str

    def __init__(self, *args: str, api_name: str=TENSORFLOW_API_NAME, v1: Optional[Sequence[str]]=None, allow_multiple_exports: bool=True):
        """Export under the names *args (first one is considered canonical).

    Args:
      *args: API names in dot delimited format.
      api_name: Name of the API you want to generate (e.g. `tensorflow` or
        `estimator`). Default is `tensorflow`.
      v1: Names for the TensorFlow V1 API. If not set, we will use V2 API names
        both for TensorFlow V1 and V2 APIs.
      allow_multiple_exports: Deprecated.
    """
        self._names = args
        self._names_v1 = v1 if v1 is not None else args
        self._api_name = api_name
        self._validate_symbol_names()

    def _validate_symbol_names(self) -> None:
        """Validate you are exporting symbols under an allowed package.

    We need to ensure things exported by tf_export, estimator_export, etc.
    export symbols under disjoint top-level package names.

    For TensorFlow, we check that it does not export anything under subpackage
    names used by components (estimator, keras, etc.).

    For each component, we check that it exports everything under its own
    subpackage.

    Raises:
      InvalidSymbolNameError: If you try to export symbol under disallowed name.
    """
        all_symbol_names = set(self._names) | set(self._names_v1)
        if self._api_name == TENSORFLOW_API_NAME:
            for subpackage in SUBPACKAGE_NAMESPACES:
                if any((n.startswith(subpackage) for n in all_symbol_names)):
                    raise InvalidSymbolNameError('@tf_export is not allowed to export symbols under %s.*' % subpackage)
        elif not all((n.startswith(self._api_name) for n in all_symbol_names)):
            raise InvalidSymbolNameError('Can only export symbols under package name of component. e.g. tensorflow_estimator must export all symbols under tf.estimator')

    def __call__(self, func: T) -> T:
        """Calls this decorator.

    Args:
      func: decorated symbol (function or class).

    Returns:
      The input function with _tf_api_names attribute set.
    """
        api_names_attr = API_ATTRS[self._api_name].names
        api_names_attr_v1 = API_ATTRS_V1[self._api_name].names
        _, undecorated_func = tf_decorator.unwrap(func)
        self.set_attr(undecorated_func, api_names_attr, self._names)
        self.set_attr(undecorated_func, api_names_attr_v1, self._names_v1)
        for name in self._names:
            _NAME_TO_SYMBOL_MAPPING[name] = func
        for name_v1 in self._names_v1:
            _NAME_TO_SYMBOL_MAPPING['compat.v1.%s' % name_v1] = func
        return func

    def set_attr(self, func: Any, api_names_attr: str, names: Sequence[str]) -> None:
        setattr(func, api_names_attr, names)

    def export_constant(self, module_name: str, name: str) -> None:
        """Store export information for constants/string literals.

    Export information is stored in the module where constants/string literals
    are defined.

    e.g.
    ```python
    foo = 1
    bar = 2
    tf_export("consts.foo").export_constant(__name__, 'foo')
    tf_export("consts.bar").export_constant(__name__, 'bar')
    ```

    Args:
      module_name: (string) Name of the module to store constant at.
      name: (string) Current constant name.
    """
        module = sys.modules[module_name]
        api_constants_attr = API_ATTRS[self._api_name].constants
        api_constants_attr_v1 = API_ATTRS_V1[self._api_name].constants
        if not hasattr(module, api_constants_attr):
            setattr(module, api_constants_attr, [])
        getattr(module, api_constants_attr).append((self._names, name))
        if not hasattr(module, api_constants_attr_v1):
            setattr(module, api_constants_attr_v1, [])
        getattr(module, api_constants_attr_v1).append((self._names_v1, name))