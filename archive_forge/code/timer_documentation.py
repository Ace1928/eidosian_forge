import time
A running stat for conveniently logging the duration of a code block.

    Example:
        wait_timer = TimerStat()
        with wait_timer:
            ray.wait(...)

    Note that this class is *not* thread-safe.
    