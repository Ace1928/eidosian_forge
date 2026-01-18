import threading
Release capacity back to the retry quota.

        The capacity being released will be truncated if necessary
        to ensure the max capacity is never exceeded.

        