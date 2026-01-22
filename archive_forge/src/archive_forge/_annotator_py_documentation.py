from . import errors
from . import graph as _mod_graph
from . import osutils, ui
Determine the single-best-revision to source for each line.

        This is meant as a compatibility thunk to how annotate() used to work.
        :return: [(ann_key, line)]
            A list of tuples with a single annotation key for each line.
        