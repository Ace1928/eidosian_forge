def __match_ancestry(self, ancestry):
    """
        Find frames matching the given ancestry.

        Returns a tuple containing the following:
          * Topmost frame matching the given ancestry or the bottom-most sentry
            frame if no frame matches.
          * Unmatched ancestry part.

        """
    stack = self.__stack
    if len(stack) == 1:
        return (stack[0], ancestry)
    previous = stack[0]
    for frame, n in zip(stack[1:], range(len(ancestry))):
        if frame.id() is not ancestry[n]:
            return (previous, ancestry[n:])
        previous = frame
    return (frame, ancestry[n + 1:])