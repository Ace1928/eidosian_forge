import functools
def unstable_test(reason):
    """Decorator used to mark test as unstable, on failure test will be skipped

    :param reason: String used in explanation, for example, 'bug 123456'.
    """

    def decor(f):

        @functools.wraps(f)
        def inner(self, *args, **kwargs):
            try:
                return f(self, *args, **kwargs)
            except Exception as e:
                msg = '%s was marked as unstable because of %s, failure was: %s' % (self.id(), reason, e)
                raise self.skipTest(msg)
        return inner
    return decor