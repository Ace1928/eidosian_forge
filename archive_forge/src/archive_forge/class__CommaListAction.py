import argparse
from osc_lib.i18n import _
class _CommaListAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(','))