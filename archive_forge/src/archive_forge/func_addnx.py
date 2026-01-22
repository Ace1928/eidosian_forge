from redis.client import NEVER_DECODE
from redis.exceptions import ModuleError
from redis.utils import HIREDIS_AVAILABLE, deprecated_function
def addnx(self, key, item):
    """
        Add an `item` to a Cuckoo Filter `key` only if item does not yet exist.
        Command might be slower that `add`.
        For more information see `CF.ADDNX <https://redis.io/commands/cf.addnx>`_.
        """
    return self.execute_command(CF_ADDNX, key, item)