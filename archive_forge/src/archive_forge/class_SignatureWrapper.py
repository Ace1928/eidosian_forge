from inspect import Parameter
from jedi.cache import memoize_method
from jedi import debug
from jedi import parser_utils
class SignatureWrapper(_SignatureMixin):

    def __init__(self, wrapped_signature):
        self._wrapped_signature = wrapped_signature

    def __getattr__(self, name):
        return getattr(self._wrapped_signature, name)