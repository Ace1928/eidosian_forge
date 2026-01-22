def _UpdateLength(self):
    """ updates our length

         *intended for internal use*

        """
    self._len = sum((len(c) for c in self.children)) + 1