import typing
import warnings
from ..api import BytesBackend
from ..api import NO_VALUE
class RedisSentinelBackend(RedisBackend):
    """A `Redis <http://redis.io/>`_ backend, using the
    `redis-py <http://pypi.python.org/pypi/redis/>`_ driver.
    This backend is to be used when using
    `Redis Sentinel <https://redis.io/docs/management/sentinel/>`_.

    .. versionadded:: 1.0.0

    Example configuration::

        from dogpile.cache import make_region

        region = make_region().configure(
            'dogpile.cache.redis_sentinel',
            arguments = {
                'sentinels': [
                    ['redis_sentinel_1', 26379],
                    ['redis_sentinel_2', 26379]
                ],
                'db': 0,
                'redis_expiration_time': 60*60*2,   # 2 hours
                'distributed_lock': True,
                'thread_local_lock': False
            }
        )


    Arguments accepted in the arguments dictionary:

    :param username: string, default is no username.

     .. versionadded:: 1.3.1

    :param password: string, default is no password.

    :param db: integer, default is ``0``.

    :param redis_expiration_time: integer, number of seconds after setting
     a value that Redis should expire it.  This should be larger than dogpile's
     cache expiration.  By default no expiration is set.

    :param distributed_lock: boolean, when True, will use a
     redis-lock as the dogpile lock. Use this when multiple processes will be
     talking to the same redis instance. When False, dogpile will
     coordinate on a regular threading mutex, Default is True.

    :param lock_timeout: integer, number of seconds after acquiring a lock that
     Redis should expire it.  This argument is only valid when
     ``distributed_lock`` is ``True``.

    :param socket_timeout: float, seconds for socket timeout.
     Default is None (no timeout).

     .. versionadded:: 1.3.2

    :param socket_connect_timeout: float, seconds for socket connection
     timeout.  Default is None (no timeout).

     .. versionadded:: 1.3.2

    :param socket_keepalive: boolean, when True, socket keepalive is enabled
     Default is False.

     .. versionadded:: 1.3.2

    :param socket_keepalive_options: dict, socket keepalive options.
     Default is {} (no options).

    :param sentinels: is a list of sentinel nodes. Each node is represented by
     a pair (hostname, port).
     Default is None (not in sentinel mode).

    :param service_name: str, the service name.
     Default is 'mymaster'.

    :param sentinel_kwargs: is a dictionary of connection arguments used when
     connecting to sentinel instances. Any argument that can be passed to
     a normal Redis connection can be specified here.
     Default is {}.

    :param connection_kwargs: dict, additional keyword arguments are passed
     along to the
     ``StrictRedis.from_url()`` method or ``StrictRedis()`` constructor
     directly, including parameters like ``ssl``, ``ssl_certfile``,
     ``charset``, etc.

    :param lock_sleep: integer, number of seconds to sleep when failed to
     acquire a lock.  This argument is only valid when
     ``distributed_lock`` is ``True``.

    :param thread_local_lock: bool, whether a thread-local Redis lock object
     should be used. This is the default, but is not compatible with
     asynchronous runners, as they run in a different thread than the one
     used to create the lock.


    """

    def __init__(self, arguments):
        arguments = arguments.copy()
        self.sentinels = arguments.pop('sentinels', None)
        self.service_name = arguments.pop('service_name', 'mymaster')
        self.sentinel_kwargs = arguments.pop('sentinel_kwargs', {})
        super().__init__(arguments={'distributed_lock': True, 'thread_local_lock': False, **arguments})

    def _imports(self):
        global redis
        import redis.sentinel

    def _create_client(self):
        sentinel_kwargs = {}
        sentinel_kwargs.update(self.sentinel_kwargs)
        sentinel_kwargs.setdefault('username', self.username)
        sentinel_kwargs.setdefault('password', self.password)
        connection_kwargs = {}
        connection_kwargs.update(self.connection_kwargs)
        connection_kwargs.setdefault('username', self.username)
        connection_kwargs.setdefault('password', self.password)
        if self.db is not None:
            connection_kwargs.setdefault('db', self.db)
            sentinel_kwargs.setdefault('db', self.db)
        if self.socket_timeout is not None:
            connection_kwargs.setdefault('socket_timeout', self.socket_timeout)
        if self.socket_connect_timeout is not None:
            connection_kwargs.setdefault('socket_connect_timeout', self.socket_connect_timeout)
        if self.socket_keepalive:
            connection_kwargs.setdefault('socket_keepalive', True)
            if self.socket_keepalive_options is not None:
                connection_kwargs.setdefault('socket_keepalive_options', self.socket_keepalive_options)
        sentinel = redis.sentinel.Sentinel(self.sentinels, sentinel_kwargs=sentinel_kwargs, **connection_kwargs)
        self.writer_client = sentinel.master_for(self.service_name)
        self.reader_client = sentinel.slave_for(self.service_name)