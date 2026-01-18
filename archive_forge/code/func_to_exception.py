import collections
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.util import traceback_utils
def to_exception(self, source_error):
    exc = self.create_exception(source_error)
    exc.__suppress_context__ = True
    exc.ag_error_metadata = self
    return exc