import os
import weakref
import torch
class EnabledProxy:
    """Stores whether the JIT is enabled or not.

    This is just a wrapper for a bool, so that we get reference semantics
    """

    def __init__(self):
        self.enabled = self.parse_env('PYTORCH_JIT', True, '> Using PyTorch JIT', '> PyTorch JIT DISABLED')

    def parse_env(self, name, default, true_message, false_message):
        value = os.environ.get(name)
        if value is None:
            return default
        if value.lower() in {'1', 'true', 'yes'}:
            return True
        elif value.lower() in {'0', 'false', 'no'}:
            return False
        if value == '1v':
            print(true_message)
            return True
        elif value == '0v':
            print(false_message)
            return False
        raise ValueError(f'Unknown setting of {name}. Try using 0 or 1.')

    def __bool__(self):
        return self.enabled