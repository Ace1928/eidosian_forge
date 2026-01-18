import abc
Try to extract a message from the given bytes.

        This is an override point for subclasses.

        This method tries to extract a message from bytes given by the
        argument.

        Raises TooSmallException if the given data is not enough to
        extract a complete message but there's still a chance to extract
        a message if more data is come later.
        