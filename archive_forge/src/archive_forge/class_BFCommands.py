from redis.client import NEVER_DECODE
from redis.exceptions import ModuleError
from redis.utils import HIREDIS_AVAILABLE, deprecated_function
class BFCommands:
    """Bloom Filter commands."""

    def create(self, key, errorRate, capacity, expansion=None, noScale=None):
        """
        Create a new Bloom Filter `key` with desired probability of false positives
        `errorRate` expected entries to be inserted as `capacity`.
        Default expansion value is 2. By default, filter is auto-scaling.
        For more information see `BF.RESERVE <https://redis.io/commands/bf.reserve>`_.
        """
        params = [key, errorRate, capacity]
        self.append_expansion(params, expansion)
        self.append_no_scale(params, noScale)
        return self.execute_command(BF_RESERVE, *params)
    reserve = create

    def add(self, key, item):
        """
        Add to a Bloom Filter `key` an `item`.
        For more information see `BF.ADD <https://redis.io/commands/bf.add>`_.
        """
        return self.execute_command(BF_ADD, key, item)

    def madd(self, key, *items):
        """
        Add to a Bloom Filter `key` multiple `items`.
        For more information see `BF.MADD <https://redis.io/commands/bf.madd>`_.
        """
        return self.execute_command(BF_MADD, key, *items)

    def insert(self, key, items, capacity=None, error=None, noCreate=None, expansion=None, noScale=None):
        """
        Add to a Bloom Filter `key` multiple `items`.

        If `nocreate` remain `None` and `key` does not exist, a new Bloom Filter
        `key` will be created with desired probability of false positives `errorRate`
        and expected entries to be inserted as `size`.
        For more information see `BF.INSERT <https://redis.io/commands/bf.insert>`_.
        """
        params = [key]
        self.append_capacity(params, capacity)
        self.append_error(params, error)
        self.append_expansion(params, expansion)
        self.append_no_create(params, noCreate)
        self.append_no_scale(params, noScale)
        self.append_items(params, items)
        return self.execute_command(BF_INSERT, *params)

    def exists(self, key, item):
        """
        Check whether an `item` exists in Bloom Filter `key`.
        For more information see `BF.EXISTS <https://redis.io/commands/bf.exists>`_.
        """
        return self.execute_command(BF_EXISTS, key, item)

    def mexists(self, key, *items):
        """
        Check whether `items` exist in Bloom Filter `key`.
        For more information see `BF.MEXISTS <https://redis.io/commands/bf.mexists>`_.
        """
        return self.execute_command(BF_MEXISTS, key, *items)

    def scandump(self, key, iter):
        """
        Begin an incremental save of the bloom filter `key`.

        This is useful for large bloom filters which cannot fit into the normal SAVE and RESTORE model.
        The first time this command is called, the value of `iter` should be 0.
        This command will return successive (iter, data) pairs until (0, NULL) to indicate completion.
        For more information see `BF.SCANDUMP <https://redis.io/commands/bf.scandump>`_.
        """
        if HIREDIS_AVAILABLE:
            raise ModuleError('This command cannot be used when hiredis is available.')
        params = [key, iter]
        options = {}
        options[NEVER_DECODE] = []
        return self.execute_command(BF_SCANDUMP, *params, **options)

    def loadchunk(self, key, iter, data):
        """
        Restore a filter previously saved using SCANDUMP.

        See the SCANDUMP command for example usage.
        This command will overwrite any bloom filter stored under key.
        Ensure that the bloom filter will not be modified between invocations.
        For more information see `BF.LOADCHUNK <https://redis.io/commands/bf.loadchunk>`_.
        """
        return self.execute_command(BF_LOADCHUNK, key, iter, data)

    def info(self, key):
        """
        Return capacity, size, number of filters, number of items inserted, and expansion rate.
        For more information see `BF.INFO <https://redis.io/commands/bf.info>`_.
        """
        return self.execute_command(BF_INFO, key)

    def card(self, key):
        """
        Returns the cardinality of a Bloom filter - number of items that were added to a Bloom filter and detected as unique
        (items that caused at least one bit to be set in at least one sub-filter).
        For more information see `BF.CARD <https://redis.io/commands/bf.card>`_.
        """
        return self.execute_command(BF_CARD, key)