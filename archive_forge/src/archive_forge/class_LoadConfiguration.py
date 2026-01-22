from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
from ncclient.operations.rpc import RPCReply
from ncclient.operations.rpc import RPCError
from ncclient import NCClientError
import math
class LoadConfiguration(RPC):

    def request(self, format='xml', action='merge', target='candidate', config=None):
        if config is not None:
            if type(config) == list:
                config = '\n'.join(config)
            if action == 'set':
                format = 'text'
            node = new_ele('load-configuration', {'action': action, 'format': format})
            if format == 'xml':
                config_node = sub_ele(node, 'configuration')
                config_node.append(config)
            if format == 'json':
                config_node = sub_ele(node, 'configuration-json').text = config
            if format == 'text' and (not action == 'set'):
                config_node = sub_ele(node, 'configuration-text').text = config
            if action == 'set' and format == 'text':
                config_node = sub_ele(node, 'configuration-set').text = config
            return self._request(node)