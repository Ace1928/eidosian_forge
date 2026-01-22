def _process_item(self, has_value, args_allowed, args_required):
    self._args_allowed += args_allowed
    self.__update_args_required_for_item(args_required)
    self.__update_has_value_for_item(has_value)