def __in_choice_context(self):
    """
        Whether we are currently processing a choice parameter group.

        This includes processing a parameter defined directly or indirectly
        within such a group.

        May only be called during parameter processing or the result will be
        calculated based on the context left behind by the previous parameter
        processing if any.

        """
    for x in self.__stack:
        if x.__class__ is ChoiceFrame:
            return True
    return False