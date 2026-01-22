import inspect
import functools
import sys
import time
class timer:
    """Decorator for timing a method call.

    Example::

        from ase.utils.timing import timer, Timer

        class A:
            def __init__(self):
                self.timer = Timer()

            @timer('Add two numbers')
            def add(self, x, y):
                return x + y

        """

    def __init__(self, name):
        self.name = name

    def __call__(self, method):
        if inspect.isgeneratorfunction(method):

            @functools.wraps(method)
            def new_method(slf, *args, **kwargs):
                gen = method(slf, *args, **kwargs)
                while True:
                    slf.timer.start(self.name)
                    try:
                        x = next(gen)
                    except StopIteration:
                        break
                    finally:
                        slf.timer.stop()
                    yield x
        else:

            @functools.wraps(method)
            def new_method(slf, *args, **kwargs):
                slf.timer.start(self.name)
                x = method(slf, *args, **kwargs)
                try:
                    slf.timer.stop()
                except IndexError:
                    pass
                return x
        return new_method