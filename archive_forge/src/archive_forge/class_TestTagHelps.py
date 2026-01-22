import argparse
import sys
from unittest import mock
from osc_lib.tests import utils as test_utils
from osc_lib.utils import tags
class TestTagHelps(test_utils.TestCase):

    def _test_tag_method_help(self, meth, exp_normal, exp_enhanced):
        """Vet the help text of the options added by the tag filtering helpers.

        :param meth: One of the ``add_tag_*`` methods.
        :param exp_normal: Expected help output without ``enhance_help``.
        :param exp_enhanced: Expected output with ``enhance_help`` set to
            ``help_enhancer``
        """
        if sys.version_info >= (3, 10):
            options_name = 'options'
        else:
            options_name = 'optional arguments'
        parser = argparse.ArgumentParser()
        meth(parser, 'test')
        self.assertEqual(exp_normal % options_name, parser.format_help())
        parser = argparse.ArgumentParser()
        meth(parser, 'test', enhance_help=help_enhancer)
        self.assertEqual(exp_enhanced % options_name, parser.format_help())

    def test_add_tag_filtering_option_to_parser(self):
        self._test_tag_method_help(tags.add_tag_filtering_option_to_parser, 'usage: run.py [-h] [--tags <tag>[,<tag>,...]] [--any-tags <tag>[,<tag>,...]]\n              [--not-tags <tag>[,<tag>,...]]\n              [--not-any-tags <tag>[,<tag>,...]]\n\n%s:\n  -h, --help            show this help message and exit\n  --tags <tag>[,<tag>,...]\n                        List test which have all given tag(s) (Comma-separated\n                        list of tags)\n  --any-tags <tag>[,<tag>,...]\n                        List test which have any given tag(s) (Comma-separated\n                        list of tags)\n  --not-tags <tag>[,<tag>,...]\n                        Exclude test which have all given tag(s) (Comma-\n                        separated list of tags)\n  --not-any-tags <tag>[,<tag>,...]\n                        Exclude test which have any given tag(s) (Comma-\n                        separated list of tags)\n', 'usage: run.py [-h] [--tags <tag>[,<tag>,...]] [--any-tags <tag>[,<tag>,...]]\n              [--not-tags <tag>[,<tag>,...]]\n              [--not-any-tags <tag>[,<tag>,...]]\n\n%s:\n  -h, --help            show this help message and exit\n  --tags <tag>[,<tag>,...]\n                        )sgat fo tsil detarapes-ammoC( )s(gat nevig lla evah\n                        hcihw tset tsiL\n  --any-tags <tag>[,<tag>,...]\n                        )sgat fo tsil detarapes-ammoC( )s(gat nevig yna evah\n                        hcihw tset tsiL\n  --not-tags <tag>[,<tag>,...]\n                        )sgat fo tsil detarapes-ammoC( )s(gat nevig lla evah\n                        hcihw tset edulcxE\n  --not-any-tags <tag>[,<tag>,...]\n                        )sgat fo tsil detarapes-ammoC( )s(gat nevig yna evah\n                        hcihw tset edulcxE\n')

    def test_add_tag_option_to_parser_for_create(self):
        self._test_tag_method_help(tags.add_tag_option_to_parser_for_create, 'usage: run.py [-h] [--tag <tag> | --no-tag]\n\n%s:\n  -h, --help   show this help message and exit\n  --tag <tag>  Tag to be added to the test (repeat option to set multiple\n               tags)\n  --no-tag     No tags associated with the test\n', 'usage: run.py [-h] [--tag <tag> | --no-tag]\n\n%s:\n  -h, --help   show this help message and exit\n  --tag <tag>  )sgat elpitlum tes ot noitpo taeper( tset eht ot dedda eb ot\n               gaT\n  --no-tag     tset eht htiw detaicossa sgat oN\n')

    def test_add_tag_option_to_parser_for_set(self):
        self._test_tag_method_help(tags.add_tag_option_to_parser_for_set, 'usage: run.py [-h] [--tag <tag>] [--no-tag]\n\n%s:\n  -h, --help   show this help message and exit\n  --tag <tag>  Tag to be added to the test (repeat option to set multiple\n               tags)\n  --no-tag     Clear tags associated with the test. Specify both --tag and\n               --no-tag to overwrite current tags\n', 'usage: run.py [-h] [--tag <tag>] [--no-tag]\n\n%s:\n  -h, --help   show this help message and exit\n  --tag <tag>  )sgat elpitlum tes ot noitpo taeper( tset eht ot dedda eb ot\n               gaT\n  --no-tag     sgat tnerruc etirwrevo ot gat-on-- dna gat-- htob yficepS .tset\n               eht htiw detaicossa sgat raelC\n')

    def test_add_tag_option_to_parser_for_unset(self):
        self._test_tag_method_help(tags.add_tag_option_to_parser_for_unset, 'usage: run.py [-h] [--tag <tag> | --all-tag]\n\n%s:\n  -h, --help   show this help message and exit\n  --tag <tag>  Tag to be removed from the test (repeat option to remove\n               multiple tags)\n  --all-tag    Clear all tags associated with the test\n', 'usage: run.py [-h] [--tag <tag> | --all-tag]\n\n%s:\n  -h, --help   show this help message and exit\n  --tag <tag>  )sgat elpitlum evomer ot noitpo taeper( tset eht morf devomer\n               eb ot gaT\n  --all-tag    tset eht htiw detaicossa sgat lla raelC\n')