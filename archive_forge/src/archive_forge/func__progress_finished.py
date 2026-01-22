import warnings
def _progress_finished(self, task):
    """Called by the ProgressTask when it finishes"""
    if not self._task_stack:
        warnings.warn('%r finished but nothing is active' % (task,))
    if task in self._task_stack:
        self._task_stack.remove(task)
    else:
        warnings.warn('%r is not in active stack %r' % (task, self._task_stack))
    if not self._task_stack:
        self._progress_all_finished()