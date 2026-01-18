import sys
from concurrent.futures import ThreadPoolExecutor
from queue import PriorityQueue

        Schedules the callable, `fn`, to be executed

        :param fn: the callable to be invoked
        :param args: the positional arguments for the callable
        :param kwargs: the keyword arguments for the callable
        :returns: a Future object representing the execution of the callable
        