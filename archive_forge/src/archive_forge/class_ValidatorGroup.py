import warnings
from collections import ChainMap
from functools import partial, partialmethod, wraps
from itertools import chain
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union, overload
from .errors import ConfigError
from .typing import AnyCallable
from .utils import ROOT_KEY, in_ipython
class ValidatorGroup:

    def __init__(self, validators: 'ValidatorListDict') -> None:
        self.validators = validators
        self.used_validators = {'*'}

    def get_validators(self, name: str) -> Optional[Dict[str, Validator]]:
        self.used_validators.add(name)
        validators = self.validators.get(name, [])
        if name != ROOT_KEY:
            validators += self.validators.get('*', [])
        if validators:
            return {getattr(v.func, '__name__', f'<No __name__: id:{id(v.func)}>'): v for v in validators}
        else:
            return None

    def check_for_unused(self) -> None:
        unused_validators = set(chain.from_iterable(((getattr(v.func, '__name__', f'<No __name__: id:{id(v.func)}>') for v in self.validators[f] if v.check_fields) for f in self.validators.keys() - self.used_validators)))
        if unused_validators:
            fn = ', '.join(unused_validators)
            raise ConfigError(f"Validators defined with incorrect fields: {fn} (use check_fields=False if you're inheriting from the model and intended this)")