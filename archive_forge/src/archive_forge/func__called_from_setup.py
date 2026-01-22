from distutils.errors import DistutilsArgError
import inspect
import glob
import platform
import distutils.command.install as orig
import setuptools
from ..warnings import SetuptoolsDeprecationWarning, SetuptoolsWarning
@staticmethod
def _called_from_setup(run_frame):
    """
        Attempt to detect whether run() was called from setup() or by another
        command.  If called by setup(), the parent caller will be the
        'run_command' method in 'distutils.dist', and *its* caller will be
        the 'run_commands' method.  If called any other way, the
        immediate caller *might* be 'run_command', but it won't have been
        called by 'run_commands'. Return True in that case or if a call stack
        is unavailable. Return False otherwise.
        """
    if run_frame is None:
        msg = 'Call stack not available. bdist_* commands may fail.'
        SetuptoolsWarning.emit(msg)
        if platform.python_implementation() == 'IronPython':
            msg = 'For best results, pass -X:Frames to enable call stack.'
            SetuptoolsWarning.emit(msg)
        return True
    frames = inspect.getouterframes(run_frame)
    for frame in frames[2:4]:
        caller, = frame[:1]
        info = inspect.getframeinfo(caller)
        caller_module = caller.f_globals.get('__name__', '')
        if caller_module == 'setuptools.dist' and info.function == 'run_command':
            continue
        return caller_module == 'distutils.dist' and info.function == 'run_commands'
    return False