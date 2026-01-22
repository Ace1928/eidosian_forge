import os
import atexit
import functools
import pickle
import sys
import time
import warnings
import numpy as np
def _get_comm():
    """Get the correct MPI world object."""
    if 'mpi4py' in sys.modules:
        return MPI4PY()
    if '_gpaw' in sys.modules:
        import _gpaw
        if hasattr(_gpaw, 'Communicator'):
            return _gpaw.Communicator()
    if '_asap' in sys.modules:
        import _asap
        if hasattr(_asap, 'Communicator'):
            return _asap.Communicator()
    return DummyMPI()