import argparse
import gc
import logging
import os
import sys
import traceback
import types
import time
import json
from pyomo.common.deprecation import deprecated
from pyomo.common.log import is_debug_set
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.fileutils import import_file
from pyomo.common.tee import capture_output
from pyomo.common.dependencies import (
from pyomo.common.collections import Bunch
from pyomo.opt import ProblemFormat
from pyomo.opt.base import SolverFactory
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.dataportal import DataPortal
from pyomo.scripting.interface import (
from pyomo.core import Model, TransformationFactory, Suffix, display
def apply_optimizer(data, instance=None):
    """
    Perform optimization with a concrete instance

    Required:
        instance:   Problem instance.

    Returned:
        results:    Optimization results.
        opt:        Optimizer object.
    """
    if not data.options.runtime.logging == 'quiet':
        sys.stdout.write('[%8.2f] Applying solver\n' % (time.time() - start_time))
        sys.stdout.flush()
    solver = data.options.solvers[0].solver_name
    if solver is None:
        raise ValueError('Problem constructing solver:  no solver specified')
    if len(data.options.solvers[0].suffixes) > 0:
        for suffix_name in data.options.solvers[0].suffixes:
            if suffix_name[0] in ['"', "'"]:
                suffix_name = suffix_name[1:-1]
            suffix = getattr(instance, suffix_name, None)
            if suffix is None:
                setattr(instance, suffix_name, Suffix(direction=Suffix.IMPORT))
            else:
                raise ValueError('Problem declaring solver suffix %s. A component with that name already exists on model %s.' % (suffix_name, instance.name))
    if getattr(data.options.solvers[0].options, 'timelimit', 0) == 0:
        data.options.solvers[0].options.timelimit = None
    results = None
    solver_mngr_name = None
    if data.options.solvers[0].manager is None:
        solver_mngr_name = 'serial'
    elif not data.options.solvers[0].manager in SolverManagerFactory:
        raise ValueError('Unknown solver manager %s' % data.options.solvers[0].manager)
    else:
        solver_mngr_name = data.options.solvers[0].manager
    solver_mngr_kwds = {}
    with SolverManagerFactory(solver_mngr_name, **solver_mngr_kwds) as solver_mngr:
        if solver_mngr is None:
            msg = "Problem constructing solver manager '%s'"
            raise ValueError(msg % str(data.options.solvers[0].manager))
        keywords = {}
        if data.options.runtime.keep_files or data.options.postsolve.print_logfile:
            keywords['keepfiles'] = True
        if data.options.model.symbolic_solver_labels:
            keywords['symbolic_solver_labels'] = True
        if data.options.model.file_determinism is not None:
            keywords['file_determinism'] = data.options.model.file_determinism
        keywords['tee'] = data.options.runtime.stream_output
        keywords['timelimit'] = getattr(data.options.solvers[0].options, 'timelimit', 0)
        keywords['report_timing'] = data.options.runtime.report_timing
        if solver_mngr_name == 'serial':
            sf_kwds = {}
            sf_kwds['solver_io'] = data.options.solvers[0].io_format
            if data.options.solvers[0].solver_executable is not None:
                sf_kwds['executable'] = data.options.solvers[0].solver_executable
            with SolverFactory(solver, **sf_kwds) as opt:
                if opt is None:
                    raise ValueError('Problem constructing solver `%s`' % str(solver))
                for name in registered_callback:
                    opt.set_callback(name, registered_callback[name])
                if len(data.options.solvers[0].options) > 0:
                    opt.set_options(data.options.solvers[0].options)
                if not data.options.solvers[0].options_string is None:
                    opt.set_options(data.options.solvers[0].options_string)
                results = solver_mngr.solve(instance, opt=opt, **keywords)
        else:
            if len(data.options.solvers[0].options) > 0 and (not data.options.solvers[0].options_string is None):
                ostring = ' '.join(('%s=%s' % (key, value) for key, value in data.options.solvers[0].options.iteritems() if not value is None))
                keywords['options'] = ostring + ' ' + data.options.solvers[0].options_string
            elif len(data.options.solvers[0].options) > 0:
                keywords['options'] = data.options.solvers[0].options
            else:
                keywords['options'] = data.options.solvers[0].options_string
            results = solver_mngr.solve(instance, opt=solver, **keywords)
    if data.options.runtime.profile_memory >= 1 and pympler_available:
        global memory_data
        mem_used = pympler.muppy.get_size(pympler.muppy.get_objects())
        if mem_used > data.local.max_memory:
            data.local.max_memory = mem_used
        print('   Total memory = %d bytes following optimization' % mem_used)
    return Bunch(results=results, opt=solver, local=data.local)