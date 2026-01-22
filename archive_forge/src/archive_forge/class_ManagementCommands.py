import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class ManagementCommands(CommandsProtocol):
    """
    Redis management commands
    """

    def auth(self, password: str, username: Optional[str]=None, **kwargs):
        """
        Authenticates the user. If you do not pass username, Redis will try to
        authenticate for the "default" user. If you do pass username, it will
        authenticate for the given user.
        For more information see https://redis.io/commands/auth
        """
        pieces = []
        if username is not None:
            pieces.append(username)
        pieces.append(password)
        return self.execute_command('AUTH', *pieces, **kwargs)

    def bgrewriteaof(self, **kwargs):
        """Tell the Redis server to rewrite the AOF file from data in memory.

        For more information see https://redis.io/commands/bgrewriteaof
        """
        return self.execute_command('BGREWRITEAOF', **kwargs)

    def bgsave(self, schedule: bool=True, **kwargs) -> ResponseT:
        """
        Tell the Redis server to save its data to disk.  Unlike save(),
        this method is asynchronous and returns immediately.

        For more information see https://redis.io/commands/bgsave
        """
        pieces = []
        if schedule:
            pieces.append('SCHEDULE')
        return self.execute_command('BGSAVE', *pieces, **kwargs)

    def role(self) -> ResponseT:
        """
        Provide information on the role of a Redis instance in
        the context of replication, by returning if the instance
        is currently a master, slave, or sentinel.

        For more information see https://redis.io/commands/role
        """
        return self.execute_command('ROLE')

    def client_kill(self, address: str, **kwargs) -> ResponseT:
        """Disconnects the client at ``address`` (ip:port)

        For more information see https://redis.io/commands/client-kill
        """
        return self.execute_command('CLIENT KILL', address, **kwargs)

    def client_kill_filter(self, _id: Union[str, None]=None, _type: Union[str, None]=None, addr: Union[str, None]=None, skipme: Union[bool, None]=None, laddr: Union[bool, None]=None, user: str=None, **kwargs) -> ResponseT:
        """
        Disconnects client(s) using a variety of filter options
        :param _id: Kills a client by its unique ID field
        :param _type: Kills a client by type where type is one of 'normal',
        'master', 'slave' or 'pubsub'
        :param addr: Kills a client by its 'address:port'
        :param skipme: If True, then the client calling the command
        will not get killed even if it is identified by one of the filter
        options. If skipme is not provided, the server defaults to skipme=True
        :param laddr: Kills a client by its 'local (bind) address:port'
        :param user: Kills a client for a specific user name
        """
        args = []
        if _type is not None:
            client_types = ('normal', 'master', 'slave', 'pubsub')
            if str(_type).lower() not in client_types:
                raise DataError(f'CLIENT KILL type must be one of {client_types!r}')
            args.extend((b'TYPE', _type))
        if skipme is not None:
            if not isinstance(skipme, bool):
                raise DataError('CLIENT KILL skipme must be a bool')
            if skipme:
                args.extend((b'SKIPME', b'YES'))
            else:
                args.extend((b'SKIPME', b'NO'))
        if _id is not None:
            args.extend((b'ID', _id))
        if addr is not None:
            args.extend((b'ADDR', addr))
        if laddr is not None:
            args.extend((b'LADDR', laddr))
        if user is not None:
            args.extend((b'USER', user))
        if not args:
            raise DataError('CLIENT KILL <filter> <value> ... ... <filter> <value> must specify at least one filter')
        return self.execute_command('CLIENT KILL', *args, **kwargs)

    def client_info(self, **kwargs) -> ResponseT:
        """
        Returns information and statistics about the current
        client connection.

        For more information see https://redis.io/commands/client-info
        """
        return self.execute_command('CLIENT INFO', **kwargs)

    def client_list(self, _type: Union[str, None]=None, client_id: List[EncodableT]=[], **kwargs) -> ResponseT:
        """
        Returns a list of currently connected clients.
        If type of client specified, only that type will be returned.

        :param _type: optional. one of the client types (normal, master,
         replica, pubsub)
        :param client_id: optional. a list of client ids

        For more information see https://redis.io/commands/client-list
        """
        args = []
        if _type is not None:
            client_types = ('normal', 'master', 'replica', 'pubsub')
            if str(_type).lower() not in client_types:
                raise DataError(f'CLIENT LIST _type must be one of {client_types!r}')
            args.append(b'TYPE')
            args.append(_type)
        if not isinstance(client_id, list):
            raise DataError('client_id must be a list')
        if client_id:
            args.append(b'ID')
            args.append(' '.join(client_id))
        return self.execute_command('CLIENT LIST', *args, **kwargs)

    def client_getname(self, **kwargs) -> ResponseT:
        """
        Returns the current connection name

        For more information see https://redis.io/commands/client-getname
        """
        return self.execute_command('CLIENT GETNAME', **kwargs)

    def client_getredir(self, **kwargs) -> ResponseT:
        """
        Returns the ID (an integer) of the client to whom we are
        redirecting tracking notifications.

        see: https://redis.io/commands/client-getredir
        """
        return self.execute_command('CLIENT GETREDIR', **kwargs)

    def client_reply(self, reply: Union[Literal['ON'], Literal['OFF'], Literal['SKIP']], **kwargs) -> ResponseT:
        """
        Enable and disable redis server replies.

        ``reply`` Must be ON OFF or SKIP,
        ON - The default most with server replies to commands
        OFF - Disable server responses to commands
        SKIP - Skip the response of the immediately following command.

        Note: When setting OFF or SKIP replies, you will need a client object
        with a timeout specified in seconds, and will need to catch the
        TimeoutError.
        The test_client_reply unit test illustrates this, and
        conftest.py has a client with a timeout.

        See https://redis.io/commands/client-reply
        """
        replies = ['ON', 'OFF', 'SKIP']
        if reply not in replies:
            raise DataError(f'CLIENT REPLY must be one of {replies!r}')
        return self.execute_command('CLIENT REPLY', reply, **kwargs)

    def client_id(self, **kwargs) -> ResponseT:
        """
        Returns the current connection id

        For more information see https://redis.io/commands/client-id
        """
        return self.execute_command('CLIENT ID', **kwargs)

    def client_tracking_on(self, clientid: Union[int, None]=None, prefix: Sequence[KeyT]=[], bcast: bool=False, optin: bool=False, optout: bool=False, noloop: bool=False) -> ResponseT:
        """
        Turn on the tracking mode.
        For more information about the options look at client_tracking func.

        See https://redis.io/commands/client-tracking
        """
        return self.client_tracking(True, clientid, prefix, bcast, optin, optout, noloop)

    def client_tracking_off(self, clientid: Union[int, None]=None, prefix: Sequence[KeyT]=[], bcast: bool=False, optin: bool=False, optout: bool=False, noloop: bool=False) -> ResponseT:
        """
        Turn off the tracking mode.
        For more information about the options look at client_tracking func.

        See https://redis.io/commands/client-tracking
        """
        return self.client_tracking(False, clientid, prefix, bcast, optin, optout, noloop)

    def client_tracking(self, on: bool=True, clientid: Union[int, None]=None, prefix: Sequence[KeyT]=[], bcast: bool=False, optin: bool=False, optout: bool=False, noloop: bool=False, **kwargs) -> ResponseT:
        """
        Enables the tracking feature of the Redis server, that is used
        for server assisted client side caching.

        ``on`` indicate for tracking on or tracking off. The dafualt is on.

        ``clientid`` send invalidation messages to the connection with
        the specified ID.

        ``bcast`` enable tracking in broadcasting mode. In this mode
        invalidation messages are reported for all the prefixes
        specified, regardless of the keys requested by the connection.

        ``optin``  when broadcasting is NOT active, normally don't track
        keys in read only commands, unless they are called immediately
        after a CLIENT CACHING yes command.

        ``optout`` when broadcasting is NOT active, normally track keys in
        read only commands, unless they are called immediately after a
        CLIENT CACHING no command.

        ``noloop`` don't send notifications about keys modified by this
        connection itself.

        ``prefix``  for broadcasting, register a given key prefix, so that
        notifications will be provided only for keys starting with this string.

        See https://redis.io/commands/client-tracking
        """
        if len(prefix) != 0 and bcast is False:
            raise DataError('Prefix can only be used with bcast')
        pieces = ['ON'] if on else ['OFF']
        if clientid is not None:
            pieces.extend(['REDIRECT', clientid])
        for p in prefix:
            pieces.extend(['PREFIX', p])
        if bcast:
            pieces.append('BCAST')
        if optin:
            pieces.append('OPTIN')
        if optout:
            pieces.append('OPTOUT')
        if noloop:
            pieces.append('NOLOOP')
        return self.execute_command('CLIENT TRACKING', *pieces)

    def client_trackinginfo(self, **kwargs) -> ResponseT:
        """
        Returns the information about the current client connection's
        use of the server assisted client side cache.

        See https://redis.io/commands/client-trackinginfo
        """
        return self.execute_command('CLIENT TRACKINGINFO', **kwargs)

    def client_setname(self, name: str, **kwargs) -> ResponseT:
        """
        Sets the current connection name

        For more information see https://redis.io/commands/client-setname

        .. note::
           This method sets client name only for **current** connection.

           If you want to set a common name for all connections managed
           by this client, use ``client_name`` constructor argument.
        """
        return self.execute_command('CLIENT SETNAME', name, **kwargs)

    def client_setinfo(self, attr: str, value: str, **kwargs) -> ResponseT:
        """
        Sets the current connection library name or version
        For mor information see https://redis.io/commands/client-setinfo
        """
        return self.execute_command('CLIENT SETINFO', attr, value, **kwargs)

    def client_unblock(self, client_id: int, error: bool=False, **kwargs) -> ResponseT:
        """
        Unblocks a connection by its client id.
        If ``error`` is True, unblocks the client with a special error message.
        If ``error`` is False (default), the client is unblocked using the
        regular timeout mechanism.

        For more information see https://redis.io/commands/client-unblock
        """
        args = ['CLIENT UNBLOCK', int(client_id)]
        if error:
            args.append(b'ERROR')
        return self.execute_command(*args, **kwargs)

    def client_pause(self, timeout: int, all: bool=True, **kwargs) -> ResponseT:
        """
        Suspend all the Redis clients for the specified amount of time.


        For more information see https://redis.io/commands/client-pause

        :param timeout: milliseconds to pause clients
        :param all: If true (default) all client commands are blocked.
        otherwise, clients are only blocked if they attempt to execute
        a write command.
        For the WRITE mode, some commands have special behavior:
        EVAL/EVALSHA: Will block client for all scripts.
        PUBLISH: Will block client.
        PFCOUNT: Will block client.
        WAIT: Acknowledgments will be delayed, so this command will
        appear blocked.
        """
        args = ['CLIENT PAUSE', str(timeout)]
        if not isinstance(timeout, int):
            raise DataError('CLIENT PAUSE timeout must be an integer')
        if not all:
            args.append('WRITE')
        return self.execute_command(*args, **kwargs)

    def client_unpause(self, **kwargs) -> ResponseT:
        """
        Unpause all redis clients

        For more information see https://redis.io/commands/client-unpause
        """
        return self.execute_command('CLIENT UNPAUSE', **kwargs)

    def client_no_evict(self, mode: str) -> Union[Awaitable[str], str]:
        """
        Sets the client eviction mode for the current connection.

        For more information see https://redis.io/commands/client-no-evict
        """
        return self.execute_command('CLIENT NO-EVICT', mode)

    def client_no_touch(self, mode: str) -> Union[Awaitable[str], str]:
        """
        # The command controls whether commands sent by the client will alter
        # the LRU/LFU of the keys they access.
        # When turned on, the current client will not change LFU/LRU stats,
        # unless it sends the TOUCH command.

        For more information see https://redis.io/commands/client-no-touch
        """
        return self.execute_command('CLIENT NO-TOUCH', mode)

    def command(self, **kwargs):
        """
        Returns dict reply of details about all Redis commands.

        For more information see https://redis.io/commands/command
        """
        return self.execute_command('COMMAND', **kwargs)

    def command_info(self, **kwargs) -> None:
        raise NotImplementedError('COMMAND INFO is intentionally not implemented in the client.')

    def command_count(self, **kwargs) -> ResponseT:
        return self.execute_command('COMMAND COUNT', **kwargs)

    def command_list(self, module: Optional[str]=None, category: Optional[str]=None, pattern: Optional[str]=None) -> ResponseT:
        """
        Return an array of the server's command names.
        You can use one of the following filters:
        ``module``: get the commands that belong to the module
        ``category``: get the commands in the ACL category
        ``pattern``: get the commands that match the given pattern

        For more information see https://redis.io/commands/command-list/
        """
        pieces = []
        if module is not None:
            pieces.extend(['MODULE', module])
        if category is not None:
            pieces.extend(['ACLCAT', category])
        if pattern is not None:
            pieces.extend(['PATTERN', pattern])
        if pieces:
            pieces.insert(0, 'FILTERBY')
        return self.execute_command('COMMAND LIST', *pieces)

    def command_getkeysandflags(self, *args: List[str]) -> List[Union[str, List[str]]]:
        """
        Returns array of keys from a full Redis command and their usage flags.

        For more information see https://redis.io/commands/command-getkeysandflags
        """
        return self.execute_command('COMMAND GETKEYSANDFLAGS', *args)

    def command_docs(self, *args):
        """
        This function throws a NotImplementedError since it is intentionally
        not supported.
        """
        raise NotImplementedError('COMMAND DOCS is intentionally not implemented in the client.')

    def config_get(self, pattern: PatternT='*', *args: List[PatternT], **kwargs) -> ResponseT:
        """
        Return a dictionary of configuration based on the ``pattern``

        For more information see https://redis.io/commands/config-get
        """
        return self.execute_command('CONFIG GET', pattern, *args, **kwargs)

    def config_set(self, name: KeyT, value: EncodableT, *args: List[Union[KeyT, EncodableT]], **kwargs) -> ResponseT:
        """Set config item ``name`` with ``value``

        For more information see https://redis.io/commands/config-set
        """
        return self.execute_command('CONFIG SET', name, value, *args, **kwargs)

    def config_resetstat(self, **kwargs) -> ResponseT:
        """
        Reset runtime statistics

        For more information see https://redis.io/commands/config-resetstat
        """
        return self.execute_command('CONFIG RESETSTAT', **kwargs)

    def config_rewrite(self, **kwargs) -> ResponseT:
        """
        Rewrite config file with the minimal change to reflect running config.

        For more information see https://redis.io/commands/config-rewrite
        """
        return self.execute_command('CONFIG REWRITE', **kwargs)

    def dbsize(self, **kwargs) -> ResponseT:
        """
        Returns the number of keys in the current database

        For more information see https://redis.io/commands/dbsize
        """
        return self.execute_command('DBSIZE', **kwargs)

    def debug_object(self, key: KeyT, **kwargs) -> ResponseT:
        """
        Returns version specific meta information about a given key

        For more information see https://redis.io/commands/debug-object
        """
        return self.execute_command('DEBUG OBJECT', key, **kwargs)

    def debug_segfault(self, **kwargs) -> None:
        raise NotImplementedError('\n            DEBUG SEGFAULT is intentionally not implemented in the client.\n\n            For more information see https://redis.io/commands/debug-segfault\n            ')

    def echo(self, value: EncodableT, **kwargs) -> ResponseT:
        """
        Echo the string back from the server

        For more information see https://redis.io/commands/echo
        """
        return self.execute_command('ECHO', value, **kwargs)

    def flushall(self, asynchronous: bool=False, **kwargs) -> ResponseT:
        """
        Delete all keys in all databases on the current host.

        ``asynchronous`` indicates whether the operation is
        executed asynchronously by the server.

        For more information see https://redis.io/commands/flushall
        """
        args = []
        if asynchronous:
            args.append(b'ASYNC')
        return self.execute_command('FLUSHALL', *args, **kwargs)

    def flushdb(self, asynchronous: bool=False, **kwargs) -> ResponseT:
        """
        Delete all keys in the current database.

        ``asynchronous`` indicates whether the operation is
        executed asynchronously by the server.

        For more information see https://redis.io/commands/flushdb
        """
        args = []
        if asynchronous:
            args.append(b'ASYNC')
        return self.execute_command('FLUSHDB', *args, **kwargs)

    def sync(self) -> ResponseT:
        """
        Initiates a replication stream from the master.

        For more information see https://redis.io/commands/sync
        """
        from redis.client import NEVER_DECODE
        options = {}
        options[NEVER_DECODE] = []
        return self.execute_command('SYNC', **options)

    def psync(self, replicationid: str, offset: int):
        """
        Initiates a replication stream from the master.
        Newer version for `sync`.

        For more information see https://redis.io/commands/sync
        """
        from redis.client import NEVER_DECODE
        options = {}
        options[NEVER_DECODE] = []
        return self.execute_command('PSYNC', replicationid, offset, **options)

    def swapdb(self, first: int, second: int, **kwargs) -> ResponseT:
        """
        Swap two databases

        For more information see https://redis.io/commands/swapdb
        """
        return self.execute_command('SWAPDB', first, second, **kwargs)

    def select(self, index: int, **kwargs) -> ResponseT:
        """Select the Redis logical database at index.

        See: https://redis.io/commands/select
        """
        return self.execute_command('SELECT', index, **kwargs)

    def info(self, section: Union[str, None]=None, *args: List[str], **kwargs) -> ResponseT:
        """
        Returns a dictionary containing information about the Redis server

        The ``section`` option can be used to select a specific section
        of information

        The section option is not supported by older versions of Redis Server,
        and will generate ResponseError

        For more information see https://redis.io/commands/info
        """
        if section is None:
            return self.execute_command('INFO', **kwargs)
        else:
            return self.execute_command('INFO', section, *args, **kwargs)

    def lastsave(self, **kwargs) -> ResponseT:
        """
        Return a Python datetime object representing the last time the
        Redis database was saved to disk

        For more information see https://redis.io/commands/lastsave
        """
        return self.execute_command('LASTSAVE', **kwargs)

    def latency_doctor(self):
        """Raise a NotImplementedError, as the client will not support LATENCY DOCTOR.
        This funcion is best used within the redis-cli.

        For more information see https://redis.io/commands/latency-doctor
        """
        raise NotImplementedError('\n            LATENCY DOCTOR is intentionally not implemented in the client.\n\n            For more information see https://redis.io/commands/latency-doctor\n            ')

    def latency_graph(self):
        """Raise a NotImplementedError, as the client will not support LATENCY GRAPH.
        This funcion is best used within the redis-cli.

        For more information see https://redis.io/commands/latency-graph.
        """
        raise NotImplementedError('\n            LATENCY GRAPH is intentionally not implemented in the client.\n\n            For more information see https://redis.io/commands/latency-graph\n            ')

    def lolwut(self, *version_numbers: Union[str, float], **kwargs) -> ResponseT:
        """
        Get the Redis version and a piece of generative computer art

        See: https://redis.io/commands/lolwut
        """
        if version_numbers:
            return self.execute_command('LOLWUT VERSION', *version_numbers, **kwargs)
        else:
            return self.execute_command('LOLWUT', **kwargs)

    def reset(self) -> ResponseT:
        """Perform a full reset on the connection's server side contenxt.

        See: https://redis.io/commands/reset
        """
        return self.execute_command('RESET')

    def migrate(self, host: str, port: int, keys: KeysT, destination_db: int, timeout: int, copy: bool=False, replace: bool=False, auth: Union[str, None]=None, **kwargs) -> ResponseT:
        """
        Migrate 1 or more keys from the current Redis server to a different
        server specified by the ``host``, ``port`` and ``destination_db``.

        The ``timeout``, specified in milliseconds, indicates the maximum
        time the connection between the two servers can be idle before the
        command is interrupted.

        If ``copy`` is True, the specified ``keys`` are NOT deleted from
        the source server.

        If ``replace`` is True, this operation will overwrite the keys
        on the destination server if they exist.

        If ``auth`` is specified, authenticate to the destination server with
        the password provided.

        For more information see https://redis.io/commands/migrate
        """
        keys = list_or_args(keys, [])
        if not keys:
            raise DataError('MIGRATE requires at least one key')
        pieces = []
        if copy:
            pieces.append(b'COPY')
        if replace:
            pieces.append(b'REPLACE')
        if auth:
            pieces.append(b'AUTH')
            pieces.append(auth)
        pieces.append(b'KEYS')
        pieces.extend(keys)
        return self.execute_command('MIGRATE', host, port, '', destination_db, timeout, *pieces, **kwargs)

    def object(self, infotype: str, key: KeyT, **kwargs) -> ResponseT:
        """
        Return the encoding, idletime, or refcount about the key
        """
        return self.execute_command('OBJECT', infotype, key, infotype=infotype, **kwargs)

    def memory_doctor(self, **kwargs) -> None:
        raise NotImplementedError('\n            MEMORY DOCTOR is intentionally not implemented in the client.\n\n            For more information see https://redis.io/commands/memory-doctor\n            ')

    def memory_help(self, **kwargs) -> None:
        raise NotImplementedError('\n            MEMORY HELP is intentionally not implemented in the client.\n\n            For more information see https://redis.io/commands/memory-help\n            ')

    def memory_stats(self, **kwargs) -> ResponseT:
        """
        Return a dictionary of memory stats

        For more information see https://redis.io/commands/memory-stats
        """
        return self.execute_command('MEMORY STATS', **kwargs)

    def memory_malloc_stats(self, **kwargs) -> ResponseT:
        """
        Return an internal statistics report from the memory allocator.

        See: https://redis.io/commands/memory-malloc-stats
        """
        return self.execute_command('MEMORY MALLOC-STATS', **kwargs)

    def memory_usage(self, key: KeyT, samples: Union[int, None]=None, **kwargs) -> ResponseT:
        """
        Return the total memory usage for key, its value and associated
        administrative overheads.

        For nested data structures, ``samples`` is the number of elements to
        sample. If left unspecified, the server's default is 5. Use 0 to sample
        all elements.

        For more information see https://redis.io/commands/memory-usage
        """
        args = []
        if isinstance(samples, int):
            args.extend([b'SAMPLES', samples])
        return self.execute_command('MEMORY USAGE', key, *args, **kwargs)

    def memory_purge(self, **kwargs) -> ResponseT:
        """
        Attempts to purge dirty pages for reclamation by allocator

        For more information see https://redis.io/commands/memory-purge
        """
        return self.execute_command('MEMORY PURGE', **kwargs)

    def latency_histogram(self, *args):
        """
        This function throws a NotImplementedError since it is intentionally
        not supported.
        """
        raise NotImplementedError('LATENCY HISTOGRAM is intentionally not implemented in the client.')

    def latency_history(self, event: str) -> ResponseT:
        """
        Returns the raw data of the ``event``'s latency spikes time series.

        For more information see https://redis.io/commands/latency-history
        """
        return self.execute_command('LATENCY HISTORY', event)

    def latency_latest(self) -> ResponseT:
        """
        Reports the latest latency events logged.

        For more information see https://redis.io/commands/latency-latest
        """
        return self.execute_command('LATENCY LATEST')

    def latency_reset(self, *events: str) -> ResponseT:
        """
        Resets the latency spikes time series of all, or only some, events.

        For more information see https://redis.io/commands/latency-reset
        """
        return self.execute_command('LATENCY RESET', *events)

    def ping(self, **kwargs) -> ResponseT:
        """
        Ping the Redis server

        For more information see https://redis.io/commands/ping
        """
        return self.execute_command('PING', **kwargs)

    def quit(self, **kwargs) -> ResponseT:
        """
        Ask the server to close the connection.

        For more information see https://redis.io/commands/quit
        """
        return self.execute_command('QUIT', **kwargs)

    def replicaof(self, *args, **kwargs) -> ResponseT:
        """
        Update the replication settings of a redis replica, on the fly.

        Examples of valid arguments include:

        NO ONE (set no replication)
        host port (set to the host and port of a redis server)

        For more information see  https://redis.io/commands/replicaof
        """
        return self.execute_command('REPLICAOF', *args, **kwargs)

    def save(self, **kwargs) -> ResponseT:
        """
        Tell the Redis server to save its data to disk,
        blocking until the save is complete

        For more information see https://redis.io/commands/save
        """
        return self.execute_command('SAVE', **kwargs)

    def shutdown(self, save: bool=False, nosave: bool=False, now: bool=False, force: bool=False, abort: bool=False, **kwargs) -> None:
        """Shutdown the Redis server.  If Redis has persistence configured,
        data will be flushed before shutdown.
        It is possible to specify modifiers to alter the behavior of the command:
        ``save`` will force a DB saving operation even if no save points are configured.
        ``nosave`` will prevent a DB saving operation even if one or more save points
        are configured.
        ``now`` skips waiting for lagging replicas, i.e. it bypasses the first step in
        the shutdown sequence.
        ``force`` ignores any errors that would normally prevent the server from exiting
        ``abort`` cancels an ongoing shutdown and cannot be combined with other flags.

        For more information see https://redis.io/commands/shutdown
        """
        if save and nosave:
            raise DataError('SHUTDOWN save and nosave cannot both be set')
        args = ['SHUTDOWN']
        if save:
            args.append('SAVE')
        if nosave:
            args.append('NOSAVE')
        if now:
            args.append('NOW')
        if force:
            args.append('FORCE')
        if abort:
            args.append('ABORT')
        try:
            self.execute_command(*args, **kwargs)
        except ConnectionError:
            return
        raise RedisError('SHUTDOWN seems to have failed.')

    def slaveof(self, host: Union[str, None]=None, port: Union[int, None]=None, **kwargs) -> ResponseT:
        """
        Set the server to be a replicated slave of the instance identified
        by the ``host`` and ``port``. If called without arguments, the
        instance is promoted to a master instead.

        For more information see https://redis.io/commands/slaveof
        """
        if host is None and port is None:
            return self.execute_command('SLAVEOF', b'NO', b'ONE', **kwargs)
        return self.execute_command('SLAVEOF', host, port, **kwargs)

    def slowlog_get(self, num: Union[int, None]=None, **kwargs) -> ResponseT:
        """
        Get the entries from the slowlog. If ``num`` is specified, get the
        most recent ``num`` items.

        For more information see https://redis.io/commands/slowlog-get
        """
        from redis.client import NEVER_DECODE
        args = ['SLOWLOG GET']
        if num is not None:
            args.append(num)
        decode_responses = self.get_connection_kwargs().get('decode_responses', False)
        if decode_responses is True:
            kwargs[NEVER_DECODE] = []
        return self.execute_command(*args, **kwargs)

    def slowlog_len(self, **kwargs) -> ResponseT:
        """
        Get the number of items in the slowlog

        For more information see https://redis.io/commands/slowlog-len
        """
        return self.execute_command('SLOWLOG LEN', **kwargs)

    def slowlog_reset(self, **kwargs) -> ResponseT:
        """
        Remove all items in the slowlog

        For more information see https://redis.io/commands/slowlog-reset
        """
        return self.execute_command('SLOWLOG RESET', **kwargs)

    def time(self, **kwargs) -> ResponseT:
        """
        Returns the server time as a 2-item tuple of ints:
        (seconds since epoch, microseconds into this second).

        For more information see https://redis.io/commands/time
        """
        return self.execute_command('TIME', **kwargs)

    def wait(self, num_replicas: int, timeout: int, **kwargs) -> ResponseT:
        """
        Redis synchronous replication
        That returns the number of replicas that processed the query when
        we finally have at least ``num_replicas``, or when the ``timeout`` was
        reached.

        For more information see https://redis.io/commands/wait
        """
        return self.execute_command('WAIT', num_replicas, timeout, **kwargs)

    def waitaof(self, num_local: int, num_replicas: int, timeout: int, **kwargs) -> ResponseT:
        """
        This command blocks the current client until all previous write
        commands by that client are acknowledged as having been fsynced
        to the AOF of the local Redis and/or at least the specified number
        of replicas.

        For more information see https://redis.io/commands/waitaof
        """
        return self.execute_command('WAITAOF', num_local, num_replicas, timeout, **kwargs)

    def hello(self):
        """
        This function throws a NotImplementedError since it is intentionally
        not supported.
        """
        raise NotImplementedError('HELLO is intentionally not implemented in the client.')

    def failover(self):
        """
        This function throws a NotImplementedError since it is intentionally
        not supported.
        """
        raise NotImplementedError('FAILOVER is intentionally not implemented in the client.')