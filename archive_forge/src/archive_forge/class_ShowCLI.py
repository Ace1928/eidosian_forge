from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
from ncclient.operations import util
class ShowCLI(RPC):

    def request(self, command=None):
        """Run CLI -show commands
        *command* (show) command to run

        """
        node = new_ele('get')
        filter = sub_ele(node, 'filter')
        block = sub_ele(filter, 'oper-data-format-cli-block')
        sub_ele(block, 'cli-show').text = command
        return self._request(node)