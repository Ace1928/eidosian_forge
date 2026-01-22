from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
class KillSession(RPC):
    """`kill-session` RPC."""

    def request(self, session_id):
        """Force the termination of a NETCONF session (not the current one!)

        *session_id* is the session identifier of the NETCONF session to be terminated as a string
        """
        node = new_ele('kill-session')
        sub_ele(node, 'session-id').text = session_id
        return self._request(node)