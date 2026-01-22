import abc
import datetime as dt
import textwrap
from osc_lib.command import command
class MistralLister(command.Lister, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def _get_format_function(self):
        raise NotImplementedError

    def get_parser(self, parsed_args):
        parser = super(MistralLister, self).get_parser(parsed_args)
        parser.add_argument('--marker', type=str, help='The last execution uuid of the previous page, displays list of executions after "marker".', default='', nargs='?')
        parser.add_argument('--limit', type=int, help='Maximum number of entries to return in a single result. ', nargs='?')
        parser.add_argument('--sort_keys', help='Comma-separated list of sort keys to sort results by. Default: created_at. Example: mistral execution-list --sort_keys=id,description', default='created_at', nargs='?')
        parser.add_argument('--sort_dirs', help='Comma-separated list of sort directions. Default: asc. Example: mistral execution-list --sort_keys=id,description --sort_dirs=asc,desc', default='asc', nargs='?')
        parser.add_argument('--filter', dest='filters', action='append', help='Filters. Can be repeated.')
        return parser

    @abc.abstractmethod
    def _get_resources(self, parsed_args):
        """Gets a list of API resources (e.g. using client)."""
        raise NotImplementedError

    def _validate_parsed_args(self, parsed_args):
        pass

    def take_action(self, parsed_args):
        self._validate_parsed_args(parsed_args)
        f = self._get_format_function()
        ret = self._get_resources(parsed_args)
        if not isinstance(ret, list):
            ret = [ret]
        data = [f(r)[1] for r in ret]
        if data:
            return (f()[0], data)
        else:
            return f()