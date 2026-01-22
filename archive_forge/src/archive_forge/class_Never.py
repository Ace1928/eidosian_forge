from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from prompt_toolkit.utils import test_callable_args
class Never(Filter):
    """
    Never enable feature.
    """

    def __call__(self, *a, **kw):
        return False

    def __invert__(self):
        return Always()