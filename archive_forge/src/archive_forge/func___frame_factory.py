def __frame_factory(self, ancestry_item):
    """Construct a new frame representing the given ancestry item."""
    frame_class = Frame
    if ancestry_item is not None and ancestry_item.choice():
        frame_class = ChoiceFrame
    return frame_class(ancestry_item, self.__error, self.__extra_parameter_errors)