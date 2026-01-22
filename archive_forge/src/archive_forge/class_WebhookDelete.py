from cliff import lister
from cliff import show
from vitrageclient.common import utils
class WebhookDelete(show.ShowOne):
    """Delete a webhook """

    def get_parser(self, prog_name):
        parser = super(WebhookDelete, self).get_parser(prog_name)
        parser.add_argument('id', help='id of webhook to delete')
        return parser

    def take_action(self, parsed_args):
        id = parsed_args.id
        result = utils.get_client(self).webhook.delete(id=id)
        return self.dict2columns(result)