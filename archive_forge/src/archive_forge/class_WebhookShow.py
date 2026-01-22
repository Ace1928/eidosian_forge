from cliff import lister
from cliff import show
from vitrageclient.common import utils
class WebhookShow(show.ShowOne):
    """Show a webhook """

    def get_parser(self, prog_name):
        parser = super(WebhookShow, self).get_parser(prog_name)
        parser.add_argument('id', help='id of webhook to show')
        return parser

    def take_action(self, parsed_args):
        id = parsed_args.id
        webhook = utils.get_client(self).webhook.show(id=id)
        return self.dict2columns(webhook)