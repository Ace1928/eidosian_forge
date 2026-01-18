from ncclient.operations.errors import OperationError
from ncclient.operations.rpc import RPC, RPCReply
from ncclient.xml_ import *
from lxml import etree
from ncclient.operations import util

        *rpc_command* specifies rpc command to be dispatched either in plain text or in xml element format (depending on command)

        *source* name of the configuration datastore being queried

        *filter* specifies the portion of the configuration to retrieve (by default entire configuration is retrieved)

        :seealso: :ref:`filter_params`

        Examples of usage::

            dispatch('clear-arp-table')

        or dispatch element like ::

            xsd_fetch = new_ele('get-xnm-information')
            sub_ele(xsd_fetch, 'type').text="xml-schema"
            sub_ele(xsd_fetch, 'namespace').text="junos-configuration"
            dispatch(xsd_fetch)
        