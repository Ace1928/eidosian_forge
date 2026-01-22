def __get_param_value(self, name):
    """
        Extract a parameter value from the remaining given arguments.

        Returns a 2-tuple consisting of the following:
          * Boolean indicating whether an argument has been specified for the
            requested input parameter.
          * Parameter value.

        """
    if self.__args:
        return (True, self.__args.pop(0))
    try:
        value = self.__kwargs.pop(name)
    except KeyError:
        return (False, None)
    return (True, value)