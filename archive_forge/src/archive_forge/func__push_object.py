import paste.util.threadinglocal as threadinglocal
def _push_object(self, obj):
    """Make ``obj`` the active object for this thread-local.

        This should be used like:

        .. code-block:: python

            obj = yourobject()
            module.glob = StackedObjectProxy()
            module.glob._push_object(obj)
            try:
                ... do stuff ...
            finally:
                module.glob._pop_object(conf)

        """
    try:
        self.____local__.objects.append(obj)
    except AttributeError:
        self.____local__.objects = []
        self.____local__.objects.append(obj)