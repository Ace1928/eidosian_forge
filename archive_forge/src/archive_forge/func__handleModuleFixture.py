import sys
from . import case
from . import util
def _handleModuleFixture(self, test, result):
    previousModule = self._get_previous_module(result)
    currentModule = test.__class__.__module__
    if currentModule == previousModule:
        return
    self._handleModuleTearDown(result)
    result._moduleSetUpFailed = False
    try:
        module = sys.modules[currentModule]
    except KeyError:
        return
    setUpModule = getattr(module, 'setUpModule', None)
    if setUpModule is not None:
        _call_if_exists(result, '_setupStdout')
        try:
            try:
                setUpModule()
            except Exception as e:
                if isinstance(result, _DebugResult):
                    raise
                result._moduleSetUpFailed = True
                self._createClassOrModuleLevelException(result, e, 'setUpModule', currentModule)
            if result._moduleSetUpFailed:
                try:
                    case.doModuleCleanups()
                except Exception as e:
                    self._createClassOrModuleLevelException(result, e, 'setUpModule', currentModule)
        finally:
            _call_if_exists(result, '_restoreStdout')