from redis.client import NEVER_DECODE
from redis.exceptions import ModuleError
from redis.utils import HIREDIS_AVAILABLE, deprecated_function
class CFCommands:
    """Cuckoo Filter commands."""

    def create(self, key, capacity, expansion=None, bucket_size=None, max_iterations=None):
        """
        Create a new Cuckoo Filter `key` an initial `capacity` items.
        For more information see `CF.RESERVE <https://redis.io/commands/cf.reserve>`_.
        """
        params = [key, capacity]
        self.append_expansion(params, expansion)
        self.append_bucket_size(params, bucket_size)
        self.append_max_iterations(params, max_iterations)
        return self.execute_command(CF_RESERVE, *params)
    reserve = create

    def add(self, key, item):
        """
        Add an `item` to a Cuckoo Filter `key`.
        For more information see `CF.ADD <https://redis.io/commands/cf.add>`_.
        """
        return self.execute_command(CF_ADD, key, item)

    def addnx(self, key, item):
        """
        Add an `item` to a Cuckoo Filter `key` only if item does not yet exist.
        Command might be slower that `add`.
        For more information see `CF.ADDNX <https://redis.io/commands/cf.addnx>`_.
        """
        return self.execute_command(CF_ADDNX, key, item)

    def insert(self, key, items, capacity=None, nocreate=None):
        """
        Add multiple `items` to a Cuckoo Filter `key`, allowing the filter
        to be created with a custom `capacity` if it does not yet exist.
        `items` must be provided as a list.
        For more information see `CF.INSERT <https://redis.io/commands/cf.insert>`_.
        """
        params = [key]
        self.append_capacity(params, capacity)
        self.append_no_create(params, nocreate)
        self.append_items(params, items)
        return self.execute_command(CF_INSERT, *params)

    def insertnx(self, key, items, capacity=None, nocreate=None):
        """
        Add multiple `items` to a Cuckoo Filter `key` only if they do not exist yet,
        allowing the filter to be created with a custom `capacity` if it does not yet exist.
        `items` must be provided as a list.
        For more information see `CF.INSERTNX <https://redis.io/commands/cf.insertnx>`_.
        """
        params = [key]
        self.append_capacity(params, capacity)
        self.append_no_create(params, nocreate)
        self.append_items(params, items)
        return self.execute_command(CF_INSERTNX, *params)

    def exists(self, key, item):
        """
        Check whether an `item` exists in Cuckoo Filter `key`.
        For more information see `CF.EXISTS <https://redis.io/commands/cf.exists>`_.
        """
        return self.execute_command(CF_EXISTS, key, item)

    def mexists(self, key, *items):
        """
        Check whether an `items` exist in Cuckoo Filter `key`.
        For more information see `CF.MEXISTS <https://redis.io/commands/cf.mexists>`_.
        """
        return self.execute_command(CF_MEXISTS, key, *items)

    def delete(self, key, item):
        """
        Delete `item` from `key`.
        For more information see `CF.DEL <https://redis.io/commands/cf.del>`_.
        """
        return self.execute_command(CF_DEL, key, item)

    def count(self, key, item):
        """
        Return the number of times an `item` may be in the `key`.
        For more information see `CF.COUNT <https://redis.io/commands/cf.count>`_.
        """
        return self.execute_command(CF_COUNT, key, item)

    def scandump(self, key, iter):
        """
        Begin an incremental save of the Cuckoo filter `key`.

        This is useful for large Cuckoo filters which cannot fit into the normal
        SAVE and RESTORE model.
        The first time this command is called, the value of `iter` should be 0.
        This command will return successive (iter, data) pairs until
        (0, NULL) to indicate completion.
        For more information see `CF.SCANDUMP <https://redis.io/commands/cf.scandump>`_.
        """
        return self.execute_command(CF_SCANDUMP, key, iter)

    def loadchunk(self, key, iter, data):
        """
        Restore a filter previously saved using SCANDUMP. See the SCANDUMP command for example usage.

        This command will overwrite any Cuckoo filter stored under key.
        Ensure that the Cuckoo filter will not be modified between invocations.
        For more information see `CF.LOADCHUNK <https://redis.io/commands/cf.loadchunk>`_.
        """
        return self.execute_command(CF_LOADCHUNK, key, iter, data)

    def info(self, key):
        """
        Return size, number of buckets, number of filter, number of items inserted,
        number of items deleted, bucket size, expansion rate, and max iteration.
        For more information see `CF.INFO <https://redis.io/commands/cf.info>`_.
        """
        return self.execute_command(CF_INFO, key)