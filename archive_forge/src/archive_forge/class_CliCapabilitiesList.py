from cliff import show
from aodhclient import utils
class CliCapabilitiesList(show.ShowOne):
    """List capabilities of alarming service"""

    def take_action(self, parsed_args):
        caps = utils.get_client(self).capabilities.list()
        return self.dict2columns(caps)