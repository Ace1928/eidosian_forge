import argparse
import textwrap
from cliff import sphinxext
from cliff.tests import base
def test_various_option_names_with_hyphen(self):
    """Handle options whose name and/or metavar contain hyphen(s)"""
    parser = argparse.ArgumentParser(prog='hello-world', add_help=False)
    parser.add_argument('--foo-bar', metavar='<foo-bar>', help='foo bar', required=True)
    parser.add_argument('--foo-bar-baz', metavar='<foo-bar-baz>', help='foo bar baz', required=True)
    parser.add_argument('--foo', metavar='<foo>', help='foo', required=True)
    parser.add_argument('--alpha', metavar='<A>', help='alpha')
    parser.add_argument('--alpha-beta', metavar='<A-B>', help='alpha beta')
    parser.add_argument('--alpha-beta-gamma', metavar='<A-B-C>', help='alpha beta gamma')
    output = '\n'.join(sphinxext._format_parser(parser))
    self.assertEqual(textwrap.dedent('\n        .. program:: hello-world\n        .. code-block:: shell\n\n            hello-world\n                --foo-bar <foo-bar>\n                --foo-bar-baz <foo-bar-baz>\n                --foo <foo>\n                [--alpha <A>]\n                [--alpha-beta <A-B>]\n                [--alpha-beta-gamma <A-B-C>]\n\n        .. option:: --foo-bar <foo-bar>\n\n            foo bar\n\n        .. option:: --foo-bar-baz <foo-bar-baz>\n\n            foo bar baz\n\n        .. option:: --foo <foo>\n\n            foo\n\n        .. option:: --alpha <A>\n\n            alpha\n\n        .. option:: --alpha-beta <A-B>\n\n            alpha beta\n\n        .. option:: --alpha-beta-gamma <A-B-C>\n\n            alpha beta gamma\n        ').lstrip(), output)