def __process_parameters(self):
    """Collect values for given web service operation input parameters."""
    for pdef in self.__param_defs:
        self.__process_parameter(*pdef)