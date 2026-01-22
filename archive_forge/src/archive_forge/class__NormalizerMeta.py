from contextlib import contextmanager
from typing import Dict, List
class _NormalizerMeta(type):

    def __new__(cls, name, bases, dct):
        new_cls = type.__new__(cls, name, bases, dct)
        new_cls.rule_value_classes = {}
        new_cls.rule_type_classes = {}
        return new_cls