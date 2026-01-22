class SimpleValidator(Validator):
    """Validator behind RegisterValidator() method.

  Validates that a single flag passes its checker function. The checker function
  takes the flag value and returns True (if value looks fine) or, if flag value
  is not valid, either returns False or raises an Exception."""

    def __init__(self, flag_name, checker, message):
        """Constructor.

    Args:
      flag_name: string, name of the flag.
      checker: function to verify the validator.
        input  - value of the corresponding flag (string, boolean, etc).
        output - Boolean. Must return True if validator constraint is satisfied.
          If constraint is not satisfied, it should either return False or
          raise Error.
      message: string, error message to be shown to the user if validator's
        condition is not satisfied
    """
        super(SimpleValidator, self).__init__(checker, message)
        self.flag_name = flag_name

    def GetFlagsNames(self):
        return [self.flag_name]

    def PrintFlagsWithValues(self, flag_values):
        return 'flag --%s=%s' % (self.flag_name, flag_values[self.flag_name].value)

    def _GetInputToCheckerFunction(self, flag_values):
        """Given flag values, construct the input to be given to checker.

    Args:
      flag_values: gflags.FlagValues
    Returns:
      value of the corresponding flag.
    """
        return flag_values[self.flag_name].value