import sys
from . import case
from . import util
def _handleClassSetUp(self, test, result):
    previousClass = getattr(result, '_previousTestClass', None)
    currentClass = test.__class__
    if currentClass == previousClass:
        return
    if result._moduleSetUpFailed:
        return
    if getattr(currentClass, '__unittest_skip__', False):
        return
    failed = False
    try:
        currentClass._classSetupFailed = False
    except TypeError:
        pass
    setUpClass = getattr(currentClass, 'setUpClass', None)
    doClassCleanups = getattr(currentClass, 'doClassCleanups', None)
    if setUpClass is not None:
        _call_if_exists(result, '_setupStdout')
        try:
            try:
                setUpClass()
            except Exception as e:
                if isinstance(result, _DebugResult):
                    raise
                failed = True
                try:
                    currentClass._classSetupFailed = True
                except TypeError:
                    pass
                className = util.strclass(currentClass)
                self._createClassOrModuleLevelException(result, e, 'setUpClass', className)
            if failed and doClassCleanups is not None:
                doClassCleanups()
                for exc_info in currentClass.tearDown_exceptions:
                    self._createClassOrModuleLevelException(result, exc_info[1], 'setUpClass', className, info=exc_info)
        finally:
            _call_if_exists(result, '_restoreStdout')