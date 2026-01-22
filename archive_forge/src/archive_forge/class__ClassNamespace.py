import types
import torch._C
class _ClassNamespace(types.ModuleType):

    def __init__(self, name):
        super().__init__('torch.classes' + name)
        self.name = name

    def __getattr__(self, attr):
        proxy = torch._C._get_custom_class_python_wrapper(self.name, attr)
        if proxy is None:
            raise RuntimeError(f'Class {self.name}.{attr} not registered!')
        return proxy