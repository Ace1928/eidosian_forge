class EventManagerBase(object):
    """Abstract class with methods :meth:`start`, :meth:`stop` and
    :meth:`dispatch` for specific class to implement.

    Example of the manager receiving touch and hover events::

        class TouchHoverManager(EventManagerBase):

            type_ids = ('touch', 'hover')

            def start(self):
                # Create additional resources, bind callbacks to self.window

            def dispatch(self, etype, me):
                if me.type_id == 'touch':
                    # Handle touch event
                elif me.type_id == 'hover'
                    # Handle hover event

            def stop(self):
                # Release resources

    """
    type_ids = None
    'Override this attribute to declare the type ids of the events which\n    manager wants to receive. This attribute will be used by\n    :class:`~kivy.core.window.WindowBase` to know which events to pass to the\n    :meth:`dispatch` method.\n\n    .. versionadded:: 2.1.0\n    '
    window = None
    'Holds the instance of the :class:`~kivy.core.window.WindowBase`.\n\n    .. versionadded:: 2.1.0\n    '

    def start(self):
        """Start the manager, bind callbacks to the objects and create
        additional resources. Attribute :attr:`window` is assigned when this
        method is called.

        .. versionadded:: 2.1.0
        """

    def dispatch(self, etype, me):
        """Dispatch event `me` to the widgets in the :attr:`window`.

        :Parameters:
            `etype`: `str`
                One of "begin", "update" or "end"
            `me`: :class:`~kivy.input.motionevent.MotionEvent`
                The Motion Event currently dispatched.
        :Returns: `bool`
            `True` to stop event dispatching

        .. versionadded:: 2.1.0
        """

    def stop(self):
        """Stop the manager, unbind from any objects and release any allocated
        resources.

        .. versionadded:: 2.1.0
        """