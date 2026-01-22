import logging
from functools import partial
from collections import defaultdict
from os.path import splitext
from .diagrams_base import BaseGraph

        Generates and saves an image of the state machine using graphviz. Note that `prog` and `args` are only part
        of the signature to mimic `Agraph.draw` and thus allow to easily switch between graph backends.
        Args:
            filename (str or file descriptor or stream or None): path and name of image output, file descriptor,
            stream object or None
            format (str): Optional format of the output file
            prog (str): ignored
            args (str): ignored
        Returns:
            None or str: Returns a binary string of the graph when the first parameter (`filename`) is set to None.
        