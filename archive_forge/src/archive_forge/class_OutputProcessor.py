from __future__ import absolute_import, division, print_function
import argparse
import itertools as it
import multiprocessing as mp
import os
import sys
from collections import MutableSequence
import numpy as np
from .utils import integer_types
class OutputProcessor(Processor):
    """
    Class for processing data and/or feeding it into some sort of output.

    """

    def process(self, data, output, **kwargs):
        """
        Processes the data and feed it to the output.

        This method must be implemented by the derived class and should
        process the given data and return the processed output.

        Parameters
        ----------
        data : depends on the implementation of subclass
            Data to be processed (e.g. written to file).
        output : str or file handle
            Output file name or file handle.
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the implementation of subclass
            Processed data.

        """
        raise NotImplementedError('Must be implemented by subclass.')