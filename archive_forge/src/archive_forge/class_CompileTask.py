import logging
import os
import sys
import taskflow.engines
from taskflow.patterns import graph_flow as gf
from taskflow import task
import example_utils as eu  # noqa
class CompileTask(task.Task):
    """Pretends to take a source and make object file."""
    default_provides = 'object_filename'

    def execute(self, source_filename):
        object_filename = '%s.o' % os.path.splitext(source_filename)[0]
        print('Compiling %s into %s' % (source_filename, object_filename))
        return object_filename