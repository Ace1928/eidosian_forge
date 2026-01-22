import argparse
import re
from IPython.core.error import UsageError
from IPython.utils.decorators import undoc
from IPython.utils.process import arg_split
from IPython.utils.text import dedent
class ArgDecorator(object):
    """ Base class for decorators to add ArgumentParser information to a method.
    """

    def __call__(self, func):
        if not getattr(func, 'has_arguments', False):
            func.has_arguments = True
            func.decorators = []
        func.decorators.append(self)
        return func

    def add_to_parser(self, parser, group):
        """ Add this object's information to the parser, if necessary.
        """
        pass