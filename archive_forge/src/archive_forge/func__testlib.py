from ._base import *
def _testlib(self):
    try:
        import transformers
    except ImportError:
        raise ValueError