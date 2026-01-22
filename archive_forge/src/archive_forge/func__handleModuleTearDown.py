import sys
from . import case
from . import util
def _handleModuleTearDown(self, result):
    previousModule = self._get_previous_module(result)
    if previousModule is None:
        return
    if result._moduleSetUpFailed:
        return
    try:
        module = sys.modules[previousModule]
    except KeyError:
        return
    _call_if_exists(result, '_setupStdout')
    try:
        tearDownModule = getattr(module, 'tearDownModule', None)
        if tearDownModule is not None:
            try:
                tearDownModule()
            except Exception as e:
                if isinstance(result, _DebugResult):
                    raise
                self._createClassOrModuleLevelException(result, e, 'tearDownModule', previousModule)
        try:
            case.doModuleCleanups()
        except Exception as e:
            if isinstance(result, _DebugResult):
                raise
            self._createClassOrModuleLevelException(result, e, 'tearDownModule', previousModule)
    finally:
        _call_if_exists(result, '_restoreStdout')