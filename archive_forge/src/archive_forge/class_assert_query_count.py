from functools import wraps
import logging
class assert_query_count(count_queries):

    def __init__(self, expected, only_select=False):
        super(assert_query_count, self).__init__(only_select=only_select)
        self.expected = expected

    def __call__(self, f):

        @wraps(f)
        def decorated(*args, **kwds):
            with self:
                ret = f(*args, **kwds)
            self._assert_count()
            return ret
        return decorated

    def _assert_count(self):
        error_msg = '%s != %s' % (self.count, self.expected)
        assert self.count == self.expected, error_msg

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(assert_query_count, self).__exit__(exc_type, exc_val, exc_tb)
        self._assert_count()