import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
class _GenePopCommandline(AbstractCommandline):
    """Return a Command Line Wrapper for GenePop (PRIVATE)."""

    def __init__(self, genepop_dir=None, cmd='Genepop', **kwargs):
        self.parameters = [_Argument(['command'], 'GenePop option to be called', is_required=True), _Argument(['mode'], 'Should always be batch', is_required=True), _Argument(['input'], 'Input file', is_required=True), _Argument(['Dememorization'], 'Dememorization step'), _Argument(['BatchNumber'], 'Number of MCMC batches'), _Argument(['BatchLength'], 'Length of MCMC chains'), _Argument(['HWtests'], 'Enumeration or MCMC'), _Argument(['IsolBDstatistic'], 'IBD statistic (a or e)'), _Argument(['MinimalDistance'], 'Minimal IBD distance'), _Argument(['GeographicScale'], 'Log or Linear')]
        AbstractCommandline.__init__(self, cmd, **kwargs)
        self.set_parameter('mode', 'Mode=Batch')

    def set_menu(self, option_list):
        """Set the menu option.

        Example set_menu([6,1]) = get all F statistics (menu 6.1)
        """
        self.set_parameter('command', 'MenuOptions=' + '.'.join((str(x) for x in option_list)))

    def set_input(self, fname):
        """Set the input file name."""
        self.set_parameter('input', 'InputFile=' + fname)