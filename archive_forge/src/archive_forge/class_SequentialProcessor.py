from __future__ import absolute_import, division, print_function
import argparse
import itertools as it
import multiprocessing as mp
import os
import sys
from collections import MutableSequence
import numpy as np
from .utils import integer_types
class SequentialProcessor(MutableSequence, Processor):
    """
    Processor class for sequential processing of data.

    Parameters
    ----------
    processors : list
         Processor instances to be processed sequentially.

    Notes
    -----
    If the `processors` list contains lists or tuples, these get wrapped as a
    SequentialProcessor itself.

    """

    def __init__(self, processors):
        self.processors = []
        for processor in processors:
            if isinstance(processor, (list, tuple)):
                processor = SequentialProcessor(processor)
            self.processors.append(processor)

    def __getitem__(self, index):
        """
        Get the Processor at the given processing chain position.

        Parameters
        ----------
        index : int
            Position inside the processing chain.

        Returns
        -------
        :class:`Processor`
            Processor at the given position.

        """
        return self.processors[index]

    def __setitem__(self, index, processor):
        """
        Set the Processor at the given processing chain position.

        Parameters
        ----------
        index : int
            Position inside the processing chain.
        processor : :class:`Processor`
            Processor to set.

        """
        self.processors[index] = processor

    def __delitem__(self, index):
        """
        Delete the Processor at the given processing chain position.

        Parameters
        ----------
        index : int
            Position inside the processing chain.

        """
        del self.processors[index]

    def __len__(self):
        """Length of the processing chain."""
        return len(self.processors)

    def insert(self, index, processor):
        """
        Insert a Processor at the given processing chain position.

        Parameters
        ----------
        index : int
            Position inside the processing chain.
        processor : :class:`Processor`
            Processor to insert.

        """
        self.processors.insert(index, processor)

    def append(self, other):
        """
        Append another Processor to the processing chain.

        Parameters
        ----------
        other : :class:`Processor`
            Processor to append to the processing chain.

        """
        self.processors.append(other)

    def extend(self, other):
        """
        Extend the processing chain with a list of Processors.

        Parameters
        ----------
        other : list
            Processors to be appended to the processing chain.

        """
        self.processors.extend(other)

    def process(self, data, **kwargs):
        """
        Process the data sequentially with the defined processing chain.

        Parameters
        ----------
        data : depends on the first processor of the processing chain
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the last processor of the processing chain
            Processed data.

        """
        for processor in self.processors:
            data = _process((processor, data, kwargs))
        return data