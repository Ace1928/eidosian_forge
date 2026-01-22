import argparse
from osc_lib.i18n import _
def add_tag_option_to_parser_for_set(parser, resource_name, enhance_help=lambda _h: _h):
    """Add tag options to a parser for set commands.

    :param parser: argparse.Argument parser object.
    :param resource_name: Description of the object being filtered.
    :param enhance_help: A callable accepting a single parameter, the
        (translated) help string, and returning a (translated) help string. May
        be used by a caller wishing to add qualifying text, such as "Applies to
        version XYZ only", to the help strings for all options produced by this
        method.
    """
    parser.add_argument('--tag', action='append', dest='tags', metavar='<tag>', help=enhance_help(_('Tag to be added to the %s (repeat option to set multiple tags)') % resource_name))
    parser.add_argument('--no-tag', action='store_true', help=enhance_help(_('Clear tags associated with the %s. Specify both --tag and --no-tag to overwrite current tags') % resource_name))