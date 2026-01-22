def addPatch(self, obj, name, value):
    """
        Add a patch so that the attribute C{name} on C{obj} will be assigned to
        C{value} when C{patch} is called or during C{runWithPatches}.

        You can restore the original values with a call to restore().
        """
    self._patchesToApply.append((obj, name, value))