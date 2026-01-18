import sys
import os
import re
import warnings
import types
import unicodedata
def walkabout(self, visitor):
    """
        Perform a tree traversal similarly to `Node.walk()` (which
        see), except also call the `dispatch_departure()` method
        before exiting each node.

        Parameter `visitor`: A `NodeVisitor` object, containing a
        ``visit`` and ``depart`` implementation for each `Node`
        subclass encountered.

        Return true if we should stop the traversal.
        """
    call_depart = True
    stop = False
    visitor.document.reporter.debug('docutils.nodes.Node.walkabout calling dispatch_visit for %s' % self.__class__.__name__)
    try:
        try:
            visitor.dispatch_visit(self)
        except SkipNode:
            return stop
        except SkipDeparture:
            call_depart = False
        children = self.children
        try:
            for child in children[:]:
                if child.walkabout(visitor):
                    stop = True
                    break
        except SkipSiblings:
            pass
    except SkipChildren:
        pass
    except StopTraversal:
        stop = True
    if call_depart:
        visitor.document.reporter.debug('docutils.nodes.Node.walkabout calling dispatch_departure for %s' % self.__class__.__name__)
        visitor.dispatch_departure(self)
    return stop