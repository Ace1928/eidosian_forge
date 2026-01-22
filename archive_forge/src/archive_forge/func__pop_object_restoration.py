import paste.util.threadinglocal as threadinglocal
def _pop_object_restoration(self, obj=None):
    if not restorer.in_restoration():
        self._pop_object_orig(obj)