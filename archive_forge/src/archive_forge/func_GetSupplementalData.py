def GetSupplementalData(self):
    if not hasattr(self, '_supplementalData'):
        self.SetSupplementalData([])
    return self._supplementalData