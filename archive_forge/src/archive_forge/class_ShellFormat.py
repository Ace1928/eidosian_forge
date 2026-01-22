import sys
from osc_lib.command import command
class ShellFormat(RawFormat):

    @property
    def formatter_default(self):
        return 'shell'