from abc import ABC
from argparse import ArgumentParser, RawTextHelpFormatter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Type
class RootOptimumCLICommand(BaseOptimumCLICommand):
    COMMAND = CommandInfo(name='root', help='optimum-cli root command')

    def __init__(self, cli_name: str, usage: Optional[str]=None, args: Optional['Namespace']=None):
        self.parser = ArgumentParser(cli_name, usage=usage)
        self.subparsers = self.parser.add_subparsers()
        self.args = None