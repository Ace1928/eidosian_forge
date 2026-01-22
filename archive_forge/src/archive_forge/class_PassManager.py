import gast as ast
import os
import re
from time import time
class PassManager(object):
    """
    Front end to the pythran pass system.
    """

    def __init__(self, module_name, module_dir=None, code=None):
        self.module_name = module_name
        self.module_dir = module_dir or os.getcwd()
        self.code = code
        self._cache = {}

    def gather(self, analysis, node, run_times=None):
        """High-level function to call an `analysis' on a `node'"""
        t0 = time()
        assert issubclass(analysis, Analysis)
        a = analysis()
        a.attach(self)
        ret = a.run(node)
        if run_times is not None:
            run_times[analysis] = run_times.get(analysis, 0) + time() - t0
        return ret

    def dump(self, backend, node):
        """High-level function to call a `backend' on a `node' to generate
        code for module `module_name'."""
        assert issubclass(backend, Backend)
        b = backend()
        b.attach(self)
        return b.run(node)

    def apply(self, transformation, node, run_times=None):
        """
        High-level function to call a `transformation' on a `node'.
        If the transformation is an analysis, the result of the analysis
        is displayed.
        """
        t0 = time()
        assert issubclass(transformation, (Transformation, Analysis))
        a = transformation()
        a.attach(self)
        ret = a.apply(node)
        if run_times is not None:
            run_times[transformation] = run_times.get(transformation, 0) + time() - t0
        return ret