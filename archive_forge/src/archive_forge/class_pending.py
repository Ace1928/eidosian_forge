import sys
import os
import re
import warnings
import types
import unicodedata
class pending(Special, Invisible, Element):
    """
    The "pending" element is used to encapsulate a pending operation: the
    operation (transform), the point at which to apply it, and any data it
    requires.  Only the pending operation's location within the document is
    stored in the public document tree (by the "pending" object itself); the
    operation and its data are stored in the "pending" object's internal
    instance attributes.

    For example, say you want a table of contents in your reStructuredText
    document.  The easiest way to specify where to put it is from within the
    document, with a directive::

        .. contents::

    But the "contents" directive can't do its work until the entire document
    has been parsed and possibly transformed to some extent.  So the directive
    code leaves a placeholder behind that will trigger the second phase of its
    processing, something like this::

        <pending ...public attributes...> + internal attributes

    Use `document.note_pending()` so that the
    `docutils.transforms.Transformer` stage of processing can run all pending
    transforms.
    """

    def __init__(self, transform, details=None, rawsource='', *children, **attributes):
        Element.__init__(self, rawsource, *children, **attributes)
        self.transform = transform
        'The `docutils.transforms.Transform` class implementing the pending\n        operation.'
        self.details = details or {}
        'Detail data (dictionary) required by the pending operation.'

    def pformat(self, indent='    ', level=0):
        internals = ['.. internal attributes:', '     .transform: %s.%s' % (self.transform.__module__, self.transform.__name__), '     .details:']
        details = list(self.details.items())
        details.sort()
        for key, value in details:
            if isinstance(value, Node):
                internals.append('%7s%s:' % ('', key))
                internals.extend(['%9s%s' % ('', line) for line in value.pformat().splitlines()])
            elif value and isinstance(value, list) and isinstance(value[0], Node):
                internals.append('%7s%s:' % ('', key))
                for v in value:
                    internals.extend(['%9s%s' % ('', line) for line in v.pformat().splitlines()])
            else:
                internals.append('%7s%s: %r' % ('', key, value))
        return Element.pformat(self, indent, level) + ''.join(['    %s%s\n' % (indent * level, line) for line in internals])

    def copy(self):
        obj = self.__class__(self.transform, self.details, self.rawsource, **self.attributes)
        obj.document = self.document
        obj.source = self.source
        obj.line = self.line
        return obj