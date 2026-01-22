from twisted.python import log
from twisted.words.xish import xpath
def addObserver(self, event, observerfn, priority=0, *args, **kwargs):
    """
        Register an observer for an event.

        Each observer will be registered with a certain priority. Higher
        priority observers get called before lower priority observers.

        @param event: Name or XPath query for the event to be monitored.
        @type event: C{str} or L{xpath.XPathQuery}.
        @param observerfn: Function to be called when the specified event
                           has been triggered. This callable takes
                           one parameter: the data object that triggered
                           the event. When specified, the C{*args} and
                           C{**kwargs} parameters to addObserver are being used
                           as additional parameters to the registered observer
                           callable.
        @param priority: (Optional) priority of this observer in relation to
                         other observer that match the same event. Defaults to
                         C{0}.
        @type priority: C{int}
        """
    self._addObserver(False, event, observerfn, priority, *args, **kwargs)