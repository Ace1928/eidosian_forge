from __future__ import annotations
import multiprocessing
import multiprocessing.process
import os
import os.path
import sys
import traceback
from typing import Any
from coverage.debug import DebugControl
class ProcessWithCoverage(OriginalProcess):
    """A replacement for multiprocess.Process that starts coverage."""

    def _bootstrap(self, *args, **kwargs):
        """Wrapper around _bootstrap to start coverage."""
        debug: DebugControl | None = None
        try:
            from coverage import Coverage
            cov = Coverage(data_suffix=True, auto_data=True)
            cov._warn_preimported_source = False
            cov.start()
            _debug = cov._debug
            assert _debug is not None
            if _debug.should('multiproc'):
                debug = _debug
            if debug:
                debug.write('Calling multiprocessing bootstrap')
        except Exception:
            print('Exception during multiprocessing bootstrap init:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise
        try:
            return original_bootstrap(self, *args, **kwargs)
        finally:
            if debug:
                debug.write('Finished multiprocessing bootstrap')
            try:
                cov.stop()
                cov.save()
            except Exception as exc:
                if debug:
                    debug.write('Exception during multiprocessing bootstrap cleanup', exc=exc)
                raise
            if debug:
                debug.write('Saved multiprocessing data')