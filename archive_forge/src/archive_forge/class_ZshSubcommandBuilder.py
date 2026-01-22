import getopt
import inspect
import itertools
from types import MethodType
from typing import Dict, List, Set
from twisted.python import reflect, usage, util
from twisted.python.compat import ioType
class ZshSubcommandBuilder(ZshBuilder):
    """
    Constructs zsh code that will complete options for a given usage.Options
    instance, and also for a single sub-command. This will only be used in
    the case where the user is completing options for a specific subcommand.

    @type subOptions: L{twisted.python.usage.Options}
    @ivar subOptions: The L{twisted.python.usage.Options} instance defined for
        the sub command.
    """

    def __init__(self, subOptions, *args):
        self.subOptions = subOptions
        ZshBuilder.__init__(self, *args)

    def write(self):
        """
        Generate the completion function and write it to the output file
        @return: L{None}
        """
        gen = ZshArgumentsGenerator(self.options, self.cmdName, self.file)
        gen.extraActions.insert(0, SubcommandAction())
        gen.write()
        gen = ZshArgumentsGenerator(self.subOptions, self.cmdName, self.file)
        gen.write()