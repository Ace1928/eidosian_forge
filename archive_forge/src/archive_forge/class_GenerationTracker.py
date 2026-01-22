import functools
import weakref
import torch.nn
from torch.nn import Module
from .utils import ExactWeakKeyDictionary, is_lazy_module
class GenerationTracker:
    generation = 0
    dynamic_classes = ExactWeakKeyDictionary()
    generation_values = ExactWeakKeyDictionary()

    @classmethod
    def tag(cls, obj):
        cls.generation_values[obj] = cls.generation

    @staticmethod
    def mark_class_dynamic(cls):
        assert issubclass(cls, torch.nn.Module)
        GenerationTracker.dynamic_classes[cls] = True

    @classmethod
    def get_generation_value(cls, obj):
        if obj not in cls.generation_values:
            return -1
        return cls.generation_values[obj]

    @classmethod
    def check(cls, obj):
        return obj in cls.generation_values and cls.generation_values[obj] == cls.generation