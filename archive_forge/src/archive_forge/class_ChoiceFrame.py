class ChoiceFrame(Frame):
    """
    _ArgParser context frame representing a choice order indicator.

    A choice requires as many input arguments as are needed to satisfy the
    least requiring of its items. For example, if we use I(n) to identify an
    item requiring n parameter, then a choice containing I(2), I(3) & I(7)
    requires 2 arguments while a choice containing I(5) & I(4) requires 4.

    Accepts an argument for each of its contained elements but allows at most
    one of its directly contained items to have a defined value.

    """

    def __init__(self, id, error, extra_parameter_errors):
        assert id.choice()
        Frame.__init__(self, id, error, extra_parameter_errors)
        self.__has_item = False

    def _process_item(self, has_value, args_allowed, args_required):
        self._args_allowed += args_allowed
        self.__update_args_required_for_item(args_required)
        self.__update_has_value_for_item(has_value)

    def __update_args_required_for_item(self, item_args_required):
        if not self.__has_item:
            self.__has_item = True
            self._args_required = item_args_required
            return
        self._args_required = min(self.args_required(), item_args_required)

    def __update_has_value_for_item(self, item_has_value):
        if item_has_value:
            if self.has_value() and self._extra_parameter_errors:
                self._error('got multiple values for a single choice parameter')
            self._has_value = True