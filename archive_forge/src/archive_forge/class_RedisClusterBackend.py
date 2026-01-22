import typing
import warnings
from ..api import BytesBackend
from ..api import NO_VALUE
class RedisClusterBackend(RedisBackend):
    """A `Redis <http://redis.io/>`_ backend, using the
    `redis-py <http://pypi.python.org/pypi/redis/>`_ driver.
    This backend is to be used when connecting to a
    `Redis Cluster <https://redis.io/docs/management/scaling/>`_ which
    will use the
    `RedisCluster Client
    <https://redis.readthedocs.io/en/stable/connections.html#cluster-client>`_.

    .. seealso::

        `Clustering <https://redis.readthedocs.io/en/stable/clustering.html>`_
        in the redis-py documentation.

    Requires redis-py version >=4.1.0.

    .. versionadded:: 1.3.2

    Connecting to the cluster requires one of:

    * Passing a list of startup nodes
    * Passing only one node of the cluster, Redis will use automatic discovery
      to find the other nodes.

    Example configuration, using startup nodes::

        from dogpile.cache import make_region
        from redis.cluster import ClusterNode

        region = make_region().configure(
            'dogpile.cache.redis_cluster',
            arguments = {
                "startup_nodes": [
                    ClusterNode('localhost', 6379),
                    ClusterNode('localhost', 6378)
                ]
            }
        )

    It is recommended to use startup nodes, so that connections will be
    successful as at least one node will always be present.  Connection
    arguments such as password, username or
    CA certificate may be passed using ``connection_kwargs``::

        from dogpile.cache import make_region
        from redis.cluster import ClusterNode

        connection_kwargs = {
            "username": "admin",
            "password": "averystrongpassword",
            "ssl": True,
            "ssl_ca_certs": "redis.pem",
        }

        nodes = [
            ClusterNode("localhost", 6379),
            ClusterNode("localhost", 6380),
            ClusterNode("localhost", 6381),
        ]

        region = make_region().configure(
            "dogpile.cache.redis_cluster",
            arguments={
                "startup_nodes": nodes,
                "connection_kwargs": connection_kwargs,
            },
        )

    Passing a URL to one node only will allow the driver to discover the whole
    cluster automatically::

        from dogpile.cache import make_region

        region = make_region().configure(
            'dogpile.cache.redis_cluster',
            arguments = {
                "url": "localhost:6379/0"
            }
        )

    A caveat of the above approach is that if the single node targeting
    is not available, this would prevent the connection from being successful.

    Parameters accepted include:

    :param startup_nodes: List of ClusterNode. The list of nodes in
     the cluster that the client will try to connect to.

    :param url: string. If provided, will override separate
     host/password/port/db params.  The format is that accepted by
     ``RedisCluster.from_url()``.

    :param db: integer, default is ``0``.

    :param redis_expiration_time: integer, number of seconds after setting
     a value that Redis should expire it.  This should be larger than dogpile's
     cache expiration.  By default no expiration is set.

    :param distributed_lock: boolean, when True, will use a
     redis-lock as the dogpile lock. Use this when multiple processes will be
     talking to the same redis instance. When left at False, dogpile will
     coordinate on a regular threading mutex.

    :param lock_timeout: integer, number of seconds after acquiring a lock that
     Redis should expire it.  This argument is only valid when
     ``distributed_lock`` is ``True``.

    :param socket_timeout: float, seconds for socket timeout.
     Default is None (no timeout).

    :param socket_connect_timeout: float, seconds for socket connection
     timeout.  Default is None (no timeout).

    :param socket_keepalive: boolean, when True, socket keepalive is enabled
     Default is False.

    :param lock_sleep: integer, number of seconds to sleep when failed to
     acquire a lock.  This argument is only valid when
     ``distributed_lock`` is ``True``.

    :param thread_local_lock: bool, whether a thread-local Redis lock object
     should be used. This is the default, but is not compatible with
     asynchronous runners, as they run in a different thread than the one
     used to create the lock.

    :param connection_kwargs: dict, additional keyword arguments are passed
     along to the
     ``RedisCluster.from_url()`` method or ``RedisCluster()`` constructor
     directly, including parameters like ``ssl``, ``ssl_certfile``,
     ``charset``, etc.

    """

    def __init__(self, arguments):
        arguments = arguments.copy()
        self.startup_nodes = arguments.pop('startup_nodes', None)
        super().__init__(arguments)

    def _imports(self):
        global redis
        import redis.cluster

    def _create_client(self):
        redis_cluster: redis.cluster.RedisCluster[typing.Any]
        if self.url is not None:
            redis_cluster = redis.cluster.RedisCluster.from_url(self.url, **self.connection_kwargs)
        else:
            redis_cluster = redis.cluster.RedisCluster(startup_nodes=self.startup_nodes, **self.connection_kwargs)
        self.writer_client = typing.cast(redis.Redis[bytes], redis_cluster)
        self.reader_client = self.writer_client