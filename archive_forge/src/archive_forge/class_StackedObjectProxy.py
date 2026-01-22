import paste.util.threadinglocal as threadinglocal
class StackedObjectProxy(object):
    """Track an object instance internally using a stack

    The StackedObjectProxy proxies access to an object internally using a
    stacked thread-local. This makes it safe for complex WSGI environments
    where access to the object may be desired in multiple places without
    having to pass the actual object around.

    New objects are added to the top of the stack with _push_object while
    objects can be removed with _pop_object.

    """

    def __init__(self, default=NoDefault, name='Default'):
        """Create a new StackedObjectProxy

        If a default is given, its used in every thread if no other object
        has been pushed on.

        """
        self.__dict__['____name__'] = name
        self.__dict__['____local__'] = threadinglocal.local()
        if default is not NoDefault:
            self.__dict__['____default_object__'] = default

    def __dir__(self):
        """Return a list of the StackedObjectProxy's and proxied
        object's (if one exists) names.
        """
        dir_list = dir(self.__class__) + list(self.__dict__.keys())
        try:
            dir_list.extend(dir(self._current_obj()))
        except TypeError:
            pass
        dir_list.sort()
        return dir_list

    def __getattr__(self, attr):
        return getattr(self._current_obj(), attr)

    def __setattr__(self, attr, value):
        setattr(self._current_obj(), attr, value)

    def __delattr__(self, name):
        delattr(self._current_obj(), name)

    def __getitem__(self, key):
        return self._current_obj()[key]

    def __setitem__(self, key, value):
        self._current_obj()[key] = value

    def __delitem__(self, key):
        del self._current_obj()[key]

    def __call__(self, *args, **kw):
        return self._current_obj()(*args, **kw)

    def __repr__(self):
        try:
            return repr(self._current_obj())
        except (TypeError, AttributeError):
            return '<%s.%s object at 0x%x>' % (self.__class__.__module__, self.__class__.__name__, id(self))

    def __iter__(self):
        return iter(self._current_obj())

    def __bool__(self):
        return bool(self._current_obj())

    def __len__(self):
        return len(self._current_obj())

    def __contains__(self, key):
        return key in self._current_obj()

    def __nonzero__(self):
        return bool(self._current_obj())

    def _current_obj(self):
        """Returns the current active object being proxied to

        In the event that no object was pushed, the default object if
        provided will be used. Otherwise, a TypeError will be raised.

        """
        try:
            objects = self.____local__.objects
        except AttributeError:
            objects = None
        if objects:
            return objects[-1]
        else:
            obj = self.__dict__.get('____default_object__', NoDefault)
            if obj is not NoDefault:
                return obj
            else:
                raise TypeError('No object (name: %s) has been registered for this thread' % self.____name__)

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

    def _pop_object(self, obj=None):
        """Remove a thread-local object.

        If ``obj`` is given, it is checked against the popped object and an
        error is emitted if they don't match.

        """
        try:
            popped = self.____local__.objects.pop()
            if obj and popped is not obj:
                raise AssertionError('The object popped (%s) is not the same as the object expected (%s)' % (popped, obj))
        except AttributeError:
            raise AssertionError('No object has been registered for this thread')

    def _object_stack(self):
        """Returns all of the objects stacked in this container

        (Might return [] if there are none)
        """
        try:
            try:
                objs = self.____local__.objects
            except AttributeError:
                return []
            return objs[:]
        except AssertionError:
            return []

    def _current_obj_restoration(self):
        request_id = restorer.in_restoration()
        if request_id:
            return restorer.get_saved_proxied_obj(self, request_id)
        return self._current_obj_orig()
    _current_obj_restoration.__doc__ = '%s\n(StackedObjectRestorer restoration enabled)' % _current_obj.__doc__

    def _push_object_restoration(self, obj):
        if not restorer.in_restoration():
            self._push_object_orig(obj)
    _push_object_restoration.__doc__ = '%s\n(StackedObjectRestorer restoration enabled)' % _push_object.__doc__

    def _pop_object_restoration(self, obj=None):
        if not restorer.in_restoration():
            self._pop_object_orig(obj)
    _pop_object_restoration.__doc__ = '%s\n(StackedObjectRestorer restoration enabled)' % _pop_object.__doc__