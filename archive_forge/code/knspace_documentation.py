from a parent namespace so that the forked namespace will contain everything
from kivy.event import EventDispatcher
from kivy.properties import StringProperty, ObjectProperty, AliasProperty
from kivy.context import register_context
Returns a new :class:`KNSpace` instance which will have access to
        all the named objects in the current namespace but will also have a
        namespace of its own that is unique to it.

        For example:

        .. code-block:: python

            forked_knspace1 = knspace.fork()
            forked_knspace2 = knspace.fork()

        Now, any names added to `knspace` will be accessible by the
        `forked_knspace1` and `forked_knspace2` namespaces by the normal means.
        However, any names added to `forked_knspace1` will not be accessible
        from `knspace` or `forked_knspace2`. Similar for `forked_knspace2`.
        