import sys
def validate_callable(func, decorator_name):
    if not hasattr(func, '__call__'):
        raise ValueError('%s is not a function. If this is a property, make sure @property appears before @%s in your source code:\n\n@property\n@%s\ndef method(...)' % (func, decorator_name, decorator_name))