import inspect
import functools
import sys
import time
Context manager for timing a block of code.

        Example (t is a timer object)::

            with t('Add two numbers'):
                x = 2 + 2

            # same as this:
            t.start('Add two numbers')
            x = 2 + 2
            t.stop()
        