import select
from pyudev._util import eintr_retry_call
Parse ``events``.

        ``events`` is a list of events as returned by
        :meth:`select.poll.poll()`.

        Yield all parsed events.

        