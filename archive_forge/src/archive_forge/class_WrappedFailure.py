import io
import os
import traceback
from oslo_utils import excutils
from oslo_utils import reflection
class WrappedFailure(Exception):
    """Wraps one or several failure objects.

    When exception/s cannot be re-raised (for example, because the value and
    traceback are lost in serialization) or there are several exceptions active
    at the same time (due to more than one thread raising exceptions), we will
    wrap the corresponding failure objects into this exception class and
    *may* reraise this exception type to allow users to handle the contained
    failures/causes as they see fit...

    See the failure class documentation for a more comprehensive set of reasons
    why this object *may* be reraised instead of the original exception.

    :param causes: the :py:class:`~taskflow.types.failure.Failure` objects
                   that caused this exception to be raised.
    """

    def __init__(self, causes):
        super(WrappedFailure, self).__init__()
        self._causes = []
        for cause in causes:
            if cause.check(type(self)) and cause.exception:
                self._causes.extend(cause.exception)
            else:
                self._causes.append(cause)

    def __iter__(self):
        """Iterate over failures that caused the exception."""
        return iter(self._causes)

    def __len__(self):
        """Return number of wrapped failures."""
        return len(self._causes)

    def check(self, *exc_classes):
        """Check if any of exception classes caused the failure/s.

        :param exc_classes: exception types/exception type names to
                            search for.

        If any of the contained failures were caused by an exception of a
        given type, the corresponding argument that matched is returned. If
        not then none is returned.
        """
        if not exc_classes:
            return None
        for cause in self:
            result = cause.check(*exc_classes)
            if result is not None:
                return result
        return None

    def __bytes__(self):
        buf = io.BytesIO()
        buf.write(b'WrappedFailure: [')
        causes_gen = (bytes(cause) for cause in self._causes)
        buf.write(b', '.join(causes_gen))
        buf.write(b']')
        return buf.getvalue()

    def __str__(self):
        buf = io.StringIO()
        buf.write(u'WrappedFailure: [')
        causes_gen = (str(cause) for cause in self._causes)
        buf.write(u', '.join(causes_gen))
        buf.write(u']')
        return buf.getvalue()