import abc
def _verify_params(self):
    """Verifies the parameters don't use any reserved parameter.

        Raises:
            ValueError: If a reserved parameter is used.
        """
    reserved_in_use = self._RESERVED_PARAMS.intersection(self.extra_params)
    if reserved_in_use:
        raise ValueError('Using a reserved parameter', reserved_in_use)