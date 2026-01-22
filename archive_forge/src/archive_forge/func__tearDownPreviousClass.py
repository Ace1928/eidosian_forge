import sys
from . import case
from . import util
def _tearDownPreviousClass(self, test, result):
    previousClass = getattr(result, '_previousTestClass', None)
    currentClass = test.__class__
    if currentClass == previousClass or previousClass is None:
        return
    if getattr(previousClass, '_classSetupFailed', False):
        return
    if getattr(result, '_moduleSetUpFailed', False):
        return
    if getattr(previousClass, '__unittest_skip__', False):
        return
    tearDownClass = getattr(previousClass, 'tearDownClass', None)
    doClassCleanups = getattr(previousClass, 'doClassCleanups', None)
    if tearDownClass is None and doClassCleanups is None:
        return
    _call_if_exists(result, '_setupStdout')
    try:
        if tearDownClass is not None:
            try:
                tearDownClass()
            except Exception as e:
                if isinstance(result, _DebugResult):
                    raise
                className = util.strclass(previousClass)
                self._createClassOrModuleLevelException(result, e, 'tearDownClass', className)
        if doClassCleanups is not None:
            doClassCleanups()
            for exc_info in previousClass.tearDown_exceptions:
                if isinstance(result, _DebugResult):
                    raise exc_info[1]
                className = util.strclass(previousClass)
                self._createClassOrModuleLevelException(result, exc_info[1], 'tearDownClass', className, info=exc_info)
    finally:
        _call_if_exists(result, '_restoreStdout')