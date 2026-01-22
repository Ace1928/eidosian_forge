from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
from ncclient.operations import util
from .errors import OperationError
import logging
class CancelCommit(RPC):
    """`cancel-commit` RPC. Depends on the `:candidate` and `:confirmed-commit` capabilities."""
    DEPENDS = [':candidate', ':confirmed-commit']

    def request(self, persist_id=None):
        """Cancel an ongoing confirmed commit. Depends on the `:candidate` and `:confirmed-commit` capabilities.

        *persist-id* value must be equal to the value given in the <persist> parameter to the previous <commit> operation.
        """
        node = new_ele('cancel-commit')
        if persist_id is not None:
            sub_ele(node, 'persist-id').text = persist_id
        return self._request(node)