from matplotlib import _api, backend_tools, cbook, widgets
def toolmanager_connect(self, s, func):
    """
        Connect event with string *s* to *func*.

        Parameters
        ----------
        s : str
            The name of the event. The following events are recognized:

            - 'tool_message_event'
            - 'tool_removed_event'
            - 'tool_added_event'

            For every tool added a new event is created

            - 'tool_trigger_TOOLNAME', where TOOLNAME is the id of the tool.

        func : callable
            Callback function for the toolmanager event with signature::

                def func(event: ToolEvent) -> Any

        Returns
        -------
        cid
            The callback id for the connection. This can be used in
            `.toolmanager_disconnect`.
        """
    return self._callbacks.connect(s, func)