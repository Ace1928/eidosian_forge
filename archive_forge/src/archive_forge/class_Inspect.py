import warnings
from billiard.common import TERM_SIGNAME
from kombu.matcher import match
from kombu.pidbox import Mailbox
from kombu.utils.compat import register_after_fork
from kombu.utils.functional import lazy
from kombu.utils.objects import cached_property
from celery.exceptions import DuplicateNodenameWarning
from celery.utils.log import get_logger
from celery.utils.text import pluralize
class Inspect:
    """API for inspecting workers.

    This class provides proxy for accessing Inspect API of workers. The API is
    defined in :py:mod:`celery.worker.control`
    """
    app = None

    def __init__(self, destination=None, timeout=1.0, callback=None, connection=None, app=None, limit=None, pattern=None, matcher=None):
        self.app = app or self.app
        self.destination = destination
        self.timeout = timeout
        self.callback = callback
        self.connection = connection
        self.limit = limit
        self.pattern = pattern
        self.matcher = matcher

    def _prepare(self, reply):
        if reply:
            by_node = flatten_reply(reply)
            if self.destination and (not isinstance(self.destination, (list, tuple))):
                return by_node.get(self.destination)
            if self.pattern:
                pattern = self.pattern
                matcher = self.matcher
                return {node: reply for node, reply in by_node.items() if match(node, pattern, matcher)}
            return by_node

    def _request(self, command, **kwargs):
        return self._prepare(self.app.control.broadcast(command, arguments=kwargs, destination=self.destination, callback=self.callback, connection=self.connection, limit=self.limit, timeout=self.timeout, reply=True, pattern=self.pattern, matcher=self.matcher))

    def report(self):
        """Return human readable report for each worker.

        Returns:
            Dict: Dictionary ``{HOSTNAME: {'ok': REPORT_STRING}}``.
        """
        return self._request('report')

    def clock(self):
        """Get the Clock value on workers.

        >>> app.control.inspect().clock()
        {'celery@node1': {'clock': 12}}

        Returns:
            Dict: Dictionary ``{HOSTNAME: CLOCK_VALUE}``.
        """
        return self._request('clock')

    def active(self, safe=None):
        """Return list of tasks currently executed by workers.

        Arguments:
            safe (Boolean): Set to True to disable deserialization.

        Returns:
            Dict: Dictionary ``{HOSTNAME: [TASK_INFO,...]}``.

        See Also:
            For ``TASK_INFO`` details see :func:`query_task` return value.

        """
        return self._request('active', safe=safe)

    def scheduled(self, safe=None):
        """Return list of scheduled tasks with details.

        Returns:
            Dict: Dictionary ``{HOSTNAME: [TASK_SCHEDULED_INFO,...]}``.

        Here is the list of ``TASK_SCHEDULED_INFO`` fields:

        * ``eta`` - scheduled time for task execution as string in ISO 8601 format
        * ``priority`` - priority of the task
        * ``request`` - field containing ``TASK_INFO`` value.

        See Also:
            For more details about ``TASK_INFO``  see :func:`query_task` return value.
        """
        return self._request('scheduled')

    def reserved(self, safe=None):
        """Return list of currently reserved tasks, not including scheduled/active.

        Returns:
            Dict: Dictionary ``{HOSTNAME: [TASK_INFO,...]}``.

        See Also:
            For ``TASK_INFO`` details see :func:`query_task` return value.
        """
        return self._request('reserved')

    def stats(self):
        """Return statistics of worker.

        Returns:
            Dict: Dictionary ``{HOSTNAME: STAT_INFO}``.

        Here is the list of ``STAT_INFO`` fields:

        * ``broker`` - Section for broker information.
            * ``connect_timeout`` - Timeout in seconds (int/float) for establishing a new connection.
            * ``heartbeat`` - Current heartbeat value (set by client).
            * ``hostname`` - Node name of the remote broker.
            * ``insist`` - No longer used.
            * ``login_method`` - Login method used to connect to the broker.
            * ``port`` - Port of the remote broker.
            * ``ssl`` - SSL enabled/disabled.
            * ``transport`` - Name of transport used (e.g., amqp or redis)
            * ``transport_options`` - Options passed to transport.
            * ``uri_prefix`` - Some transports expects the host name to be a URL.
              E.g. ``redis+socket:///tmp/redis.sock``.
              In this example the URI-prefix will be redis.
            * ``userid`` - User id used to connect to the broker with.
            * ``virtual_host`` - Virtual host used.
        * ``clock`` - Value of the workers logical clock. This is a positive integer
          and should be increasing every time you receive statistics.
        * ``uptime`` - Numbers of seconds since the worker controller was started
        * ``pid`` - Process id of the worker instance (Main process).
        * ``pool`` - Pool-specific section.
            * ``max-concurrency`` - Max number of processes/threads/green threads.
            * ``max-tasks-per-child`` - Max number of tasks a thread may execute before being recycled.
            * ``processes`` - List of PIDs (or thread-id’s).
            * ``put-guarded-by-semaphore`` - Internal
            * ``timeouts`` - Default values for time limits.
            * ``writes`` - Specific to the prefork pool, this shows the distribution
              of writes to each process in the pool when using async I/O.
        * ``prefetch_count`` - Current prefetch count value for the task consumer.
        * ``rusage`` - System usage statistics. The fields available may be different on your platform.
          From :manpage:`getrusage(2)`:

            * ``stime`` - Time spent in operating system code on behalf of this process.
            * ``utime`` - Time spent executing user instructions.
            * ``maxrss`` - The maximum resident size used by this process (in kilobytes).
            * ``idrss`` - Amount of non-shared memory used for data (in kilobytes times
              ticks of execution)
            * ``isrss`` - Amount of non-shared memory used for stack space
              (in kilobytes times ticks of execution)
            * ``ixrss`` - Amount of memory shared with other processes
              (in kilobytes times ticks of execution).
            * ``inblock`` - Number of times the file system had to read from the disk
              on behalf of this process.
            * ``oublock`` - Number of times the file system has to write to disk
              on behalf of this process.
            * ``majflt`` - Number of page faults that were serviced by doing I/O.
            * ``minflt`` - Number of page faults that were serviced without doing I/O.
            * ``msgrcv`` - Number of IPC messages received.
            * ``msgsnd`` - Number of IPC messages sent.
            * ``nvcsw`` - Number of times this process voluntarily invoked a context switch.
            * ``nivcsw`` - Number of times an involuntary context switch took place.
            * ``nsignals`` - Number of signals received.
            * ``nswap`` - The number of times this process was swapped entirely
              out of memory.
        * ``total`` - Map of task names and the total number of tasks with that type
          the worker has accepted since start-up.
        """
        return self._request('stats')

    def revoked(self):
        """Return list of revoked tasks.

        >>> app.control.inspect().revoked()
        {'celery@node1': ['16f527de-1c72-47a6-b477-c472b92fef7a']}

        Returns:
            Dict: Dictionary ``{HOSTNAME: [TASK_ID, ...]}``.
        """
        return self._request('revoked')

    def registered(self, *taskinfoitems):
        """Return all registered tasks per worker.

        >>> app.control.inspect().registered()
        {'celery@node1': ['task1', 'task1']}
        >>> app.control.inspect().registered('serializer', 'max_retries')
        {'celery@node1': ['task_foo [serializer=json max_retries=3]', 'tasb_bar [serializer=json max_retries=3]']}

        Arguments:
            taskinfoitems (Sequence[str]): List of :class:`~celery.app.task.Task`
                                           attributes to include.

        Returns:
            Dict: Dictionary ``{HOSTNAME: [TASK1_INFO, ...]}``.
        """
        return self._request('registered', taskinfoitems=taskinfoitems)
    registered_tasks = registered

    def ping(self, destination=None):
        """Ping all (or specific) workers.

        >>> app.control.inspect().ping()
        {'celery@node1': {'ok': 'pong'}, 'celery@node2': {'ok': 'pong'}}
        >>> app.control.inspect().ping(destination=['celery@node1'])
        {'celery@node1': {'ok': 'pong'}}

        Arguments:
            destination (List): If set, a list of the hosts to send the
                command to, when empty broadcast to all workers.

        Returns:
            Dict: Dictionary ``{HOSTNAME: {'ok': 'pong'}}``.

        See Also:
            :meth:`broadcast` for supported keyword arguments.
        """
        if destination:
            self.destination = destination
        return self._request('ping')

    def active_queues(self):
        """Return information about queues from which worker consumes tasks.

        Returns:
            Dict: Dictionary ``{HOSTNAME: [QUEUE_INFO, QUEUE_INFO,...]}``.

        Here is the list of ``QUEUE_INFO`` fields:

        * ``name``
        * ``exchange``
            * ``name``
            * ``type``
            * ``arguments``
            * ``durable``
            * ``passive``
            * ``auto_delete``
            * ``delivery_mode``
            * ``no_declare``
        * ``routing_key``
        * ``queue_arguments``
        * ``binding_arguments``
        * ``consumer_arguments``
        * ``durable``
        * ``exclusive``
        * ``auto_delete``
        * ``no_ack``
        * ``alias``
        * ``bindings``
        * ``no_declare``
        * ``expires``
        * ``message_ttl``
        * ``max_length``
        * ``max_length_bytes``
        * ``max_priority``

        See Also:
            See the RabbitMQ/AMQP documentation for more details about
            ``queue_info`` fields.
        Note:
            The ``queue_info`` fields are RabbitMQ/AMQP oriented.
            Not all fields applies for other transports.
        """
        return self._request('active_queues')

    def query_task(self, *ids):
        """Return detail of tasks currently executed by workers.

        Arguments:
            *ids (str): IDs of tasks to be queried.

        Returns:
            Dict: Dictionary ``{HOSTNAME: {TASK_ID: [STATE, TASK_INFO]}}``.

        Here is the list of ``TASK_INFO`` fields:
            * ``id`` - ID of the task
            * ``name`` - Name of the task
            * ``args`` - Positinal arguments passed to the task
            * ``kwargs`` - Keyword arguments passed to the task
            * ``type`` - Type of the task
            * ``hostname`` - Hostname of the worker processing the task
            * ``time_start`` - Time of processing start
            * ``acknowledged`` - True when task was acknowledged to broker
            * ``delivery_info`` - Dictionary containing delivery information
                * ``exchange`` - Name of exchange where task was published
                * ``routing_key`` - Routing key used when task was published
                * ``priority`` - Priority used when task was published
                * ``redelivered`` - True if the task was redelivered
            * ``worker_pid`` - PID of worker processing the task

        """
        if len(ids) == 1 and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return self._request('query_task', ids=ids)

    def conf(self, with_defaults=False):
        """Return configuration of each worker.

        Arguments:
            with_defaults (bool): if set to True, method returns also
                                   configuration options with default values.

        Returns:
            Dict: Dictionary ``{HOSTNAME: WORKER_CONFIGURATION}``.

        See Also:
            ``WORKER_CONFIGURATION`` is a dictionary containing current configuration options.
            See :ref:`configuration` for possible values.
        """
        return self._request('conf', with_defaults=with_defaults)

    def hello(self, from_node, revoked=None):
        return self._request('hello', from_node=from_node, revoked=revoked)

    def memsample(self):
        """Return sample current RSS memory usage.

        Note:
            Requires the psutils library.
        """
        return self._request('memsample')

    def memdump(self, samples=10):
        """Dump statistics of previous memsample requests.

        Note:
            Requires the psutils library.
        """
        return self._request('memdump', samples=samples)

    def objgraph(self, type='Request', n=200, max_depth=10):
        """Create graph of uncollected objects (memory-leak debugging).

        Arguments:
            n (int): Max number of objects to graph.
            max_depth (int): Traverse at most n levels deep.
            type (str): Name of object to graph.  Default is ``"Request"``.

        Returns:
            Dict: Dictionary ``{'filename': FILENAME}``

        Note:
            Requires the objgraph library.
        """
        return self._request('objgraph', num=n, max_depth=max_depth, type=type)