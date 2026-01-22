import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
class FunctionRegistry(type):

    def __init__(cls, name, bases, attrs):
        cls._populate_function_table()
        super(FunctionRegistry, cls).__init__(name, bases, attrs)

    def _populate_function_table(cls):
        function_table = {}
        for name, method in get_methods(cls):
            if not name.startswith('_func_'):
                continue
            signature = getattr(method, 'signature', None)
            if signature is not None:
                function_table[name[6:]] = {'function': method, 'signature': signature}
        cls.FUNCTION_TABLE = function_table