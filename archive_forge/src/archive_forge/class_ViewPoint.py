import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
class ViewPoint(Referenceable):
    """
    I act as an indirect reference to an object accessed through a
    L{pb.IPerspective}.

    Simply put, I combine an object with a perspective so that when a
    peer calls methods on the object I refer to, the method will be
    invoked with that perspective as a first argument, so that it can
    know who is calling it.

    While L{Viewable} objects will be converted to ViewPoints by default
    when they are returned from or sent as arguments to a remote
    method, any object may be manually proxied as well. (XXX: Now that
    this class is no longer named C{Proxy}, this is the only occurrence
    of the term 'proxied' in this docstring, and may be unclear.)

    This can be useful when dealing with L{pb.IPerspective}s, L{Copyable}s,
    and L{Cacheable}s.  It is legal to implement a method as such on
    a perspective::

     | def perspective_getViewPointForOther(self, name):
     |     defr = self.service.getPerspectiveRequest(name)
     |     defr.addCallbacks(lambda x, self=self: ViewPoint(self, x), log.msg)
     |     return defr

    This will allow you to have references to Perspective objects in two
    different ways.  One is through the initial 'attach' call -- each
    peer will have a L{pb.RemoteReference} to their perspective directly.  The
    other is through this method; each peer can get a L{pb.RemoteReference} to
    all other perspectives in the service; but that L{pb.RemoteReference} will
    be to a L{ViewPoint}, not directly to the object.

    The practical offshoot of this is that you can implement 2 varieties
    of remotely callable methods on this Perspective; view_xxx and
    C{perspective_xxx}. C{view_xxx} methods will follow the rules for
    ViewPoint methods (see ViewPoint.L{remoteMessageReceived}), and
    C{perspective_xxx} methods will follow the rules for Perspective
    methods.
    """

    def __init__(self, perspective, object):
        """Initialize me with a Perspective and an Object."""
        self.perspective = perspective
        self.object = object

    def processUniqueID(self):
        """Return an ID unique to a proxy for this perspective+object combination."""
        return (id(self.perspective), id(self.object))

    def remoteMessageReceived(self, broker, message, args, kw):
        """A remote message has been received.  Dispatch it appropriately.

        The default implementation is to dispatch to a method called
        'C{view_messagename}' to my Object and call it on my object with
        the same arguments, modified by inserting my Perspective as
        the first argument.
        """
        args = broker.unserialize(args, self.perspective)
        kw = broker.unserialize(kw, self.perspective)
        if not isinstance(message, str):
            message = message.decode('utf8')
        method = getattr(self.object, 'view_%s' % message)
        try:
            state = method(*(self.perspective,) + args, **kw)
        except TypeError:
            log.msg(f"{method} didn't accept {args} and {kw}")
            raise
        rv = broker.serialize(state, self.perspective, method, args, kw)
        return rv