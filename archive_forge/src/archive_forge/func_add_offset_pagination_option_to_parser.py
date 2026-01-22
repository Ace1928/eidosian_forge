from osc_lib.cli import parseractions
from openstackclient.i18n import _
def add_offset_pagination_option_to_parser(parser):
    """Add offset-based pagination options to the parser.

    APIs that use offset-based paging use the offset and limit query parameters
    to paginate through items in a collection.

    Offset-based pagination is often used where the list of items is of a fixed
    and predetermined length.
    """
    parser.add_argument('--limit', metavar='<limit>', type=int, action=parseractions.NonNegativeAction, help=_('The maximum number of entries to return. If the value exceeds the server-defined maximum, then the maximum value will be used.'))
    parser.add_argument('--offset', metavar='<offset>', type=int, action=parseractions.NonNegativeAction, default=None, help=_('The (zero-based) offset of the first item in the collection to return.'))