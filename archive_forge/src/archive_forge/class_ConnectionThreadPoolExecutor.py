import sys
from concurrent.futures import ThreadPoolExecutor
from queue import PriorityQueue
class ConnectionThreadPoolExecutor(ThreadPoolExecutor):
    """
    A wrapper class to maintain a pool of connections alongside the thread
    pool. We start by creating a priority queue of connections, and each job
    submitted takes one of those connections (initialising if necessary) and
    passes it as the first arg to the executed function.

    At the end of execution that connection is returned to the queue.

    By using a PriorityQueue we avoid creating more connections than required.
    We will only create as many connections as are required concurrently.
    """

    def __init__(self, create_connection, max_workers):
        """
        Initializes a new ThreadPoolExecutor instance.

        :param create_connection: callable to use to create new connections
        :param max_workers: the maximum number of threads that can be used
        """
        self._connections = PriorityQueue()
        self._create_connection = create_connection
        for p in range(0, max_workers):
            self._connections.put((p, None))
        super(ConnectionThreadPoolExecutor, self).__init__(max_workers)

    def submit(self, fn, *args, **kwargs):
        """
        Schedules the callable, `fn`, to be executed

        :param fn: the callable to be invoked
        :param args: the positional arguments for the callable
        :param kwargs: the keyword arguments for the callable
        :returns: a Future object representing the execution of the callable
        """

        def conn_fn():
            priority = None
            conn = None
            try:
                priority, conn = self._connections.get()
                if conn is None:
                    conn = self._create_connection()
                conn_args = (conn,) + args
                return fn(*conn_args, **kwargs)
            finally:
                if priority is not None:
                    self._connections.put((priority, conn))
        return super(ConnectionThreadPoolExecutor, self).submit(conn_fn)