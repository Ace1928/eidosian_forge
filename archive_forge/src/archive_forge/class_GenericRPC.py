from threading import Event, Lock
from uuid import uuid4
from ncclient.xml_ import *
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.transport import SessionListener
from ncclient.operations import util
from ncclient.operations.errors import OperationError, TimeoutExpiredError, MissingCapabilityError
import logging
class GenericRPC(RPC):
    """Generic rpc commands wrapper"""
    REPLY_CLS = RPCReply
    'See :class:`RPCReply`.'

    def request(self, rpc_command, source=None, filter=None, config=None, target=None, format=None):
        """
        *rpc_command* specifies rpc command to be dispatched either in plain text or in xml element format (depending on command)

        *target* name of the configuration datastore being edited

        *source* name of the configuration datastore being queried

        *config* is the configuration, which must be rooted in the `config` element. It can be specified either as a string or an :class:`~xml.etree.ElementTree.Element`.

        *filter* specifies the portion of the configuration to retrieve (by default entire configuration is retrieved)

        :seealso: :ref:`filter_params`

        Examples of usage::

            m.rpc('rpc_command')

        or dispatch element like ::

            rpc_command = new_ele('get-xnm-information')
            sub_ele(rpc_command, 'type').text = "xml-schema"
            m.rpc(rpc_command)
        """
        if etree.iselement(rpc_command):
            node = rpc_command
        else:
            node = new_ele(rpc_command)
        if target is not None:
            node.append(util.datastore_or_url('target', target, self._assert))
        if source is not None:
            node.append(util.datastore_or_url('source', source, self._assert))
        if filter is not None:
            node.append(util.build_filter(filter))
        if config is not None:
            node.append(validated_element(config, ('config', qualify('config'))))
        return self._request(node)