import logging
import os
import re
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.opt import SolverFactory, SolverManagerFactory, OptSolver
from pyomo.opt.parallel.manager import ActionManagerError, ActionStatus
from pyomo.opt.parallel.async_solver import AsynchronousSolverManager
from pyomo.core.base import Block
import pyomo.neos.kestrel
def _perform_wait_any(self):
    """
        Perform the wait_any operation.  This method returns an
        ActionHandle with the results of waiting.  If None is returned
        then the ActionManager assumes that it can call this method again.
        Note that an ActionHandle can be returned with a dummy value,
        to indicate an error.
        """
    for jobNumber in self._ah:
        status = self.kestrel.neos.getJobStatus(jobNumber, self._ah[jobNumber].password)
        if status not in ('Running', 'Waiting'):
            ah = self._ah[jobNumber]
            del self._ah[jobNumber]
            ah.status = ActionStatus.done
            opt, smap_id, load_solutions, select_index, default_variable_value = self._opt_data[jobNumber]
            del self._opt_data[jobNumber]
            args = self._args[jobNumber]
            del self._args[jobNumber]
            results = self.kestrel.neos.getFinalResults(jobNumber, ah.password)
            current_offset, current_message = self._neos_log[jobNumber]
            with open(opt._log_file, 'w') as OUTPUT:
                OUTPUT.write(current_message)
            with open(opt._soln_file, 'w') as OUTPUT:
                OUTPUT.write(results.data.decode('utf-8'))
            rc = None
            try:
                solver_results = opt.process_output(rc)
            except:
                _neos_error('Error parsing NEOS solution file', results, current_message)
                return ah
            solver_results._smap_id = smap_id
            self.results[ah.id] = solver_results
            if isinstance(args[0], Block):
                _model = args[0]
                if load_solutions:
                    try:
                        _model.solutions.load_from(solver_results, select=select_index, default_variable_value=default_variable_value)
                    except:
                        _neos_error('Error loading NEOS solution into model', results, current_message)
                    solver_results._smap_id = None
                    solver_results.solution.clear()
                else:
                    solver_results._smap = _model.solutions.symbol_map[smap_id]
                    _model.solutions.delete_symbol_map(smap_id)
            return ah
        else:
            current_offset, current_message = self._neos_log[jobNumber]
            try:
                message_fragment, new_offset = self.kestrel.neos.getIntermediateResults(jobNumber, self._ah[jobNumber].password, current_offset)
                logger.info(message_fragment)
                self._neos_log[jobNumber] = (new_offset, current_message + message_fragment.data.decode('utf-8'))
            except xmlrpc_client.ProtocolError:
                pass
    return None