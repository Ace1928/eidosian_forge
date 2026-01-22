def __update_args_required_for_item(self, item_args_required):
    if not self.__has_item:
        self.__has_item = True
        self._args_required = item_args_required
        return
    self._args_required = min(self.args_required(), item_args_required)