import pkg_resources
import argparse
import logging
import sys
from warnings import warn
class HelpfulArgumentParser(argparse.ArgumentParser):

    def error(self, message):
        """error(message: string)

        Prints a usage message incorporating the message to stderr and
        exits.

        If you override this in a subclass, it should not return -- it
        should either exit or raise an exception.
        """
        self.print_help(sys.stderr)
        self._print_message('\n')
        self.exit(2, '%s: %s\n' % (self.prog, message))