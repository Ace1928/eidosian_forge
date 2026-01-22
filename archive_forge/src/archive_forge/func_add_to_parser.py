import argparse
import re
from IPython.core.error import UsageError
from IPython.utils.decorators import undoc
from IPython.utils.process import arg_split
from IPython.utils.text import dedent
def add_to_parser(self, parser, group):
    """ Add this object's information to the parser.
        """
    return parser.add_argument_group(*self.args, **self.kwds)