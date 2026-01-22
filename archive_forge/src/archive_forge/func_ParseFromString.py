def ParseFromString(self, serialized):
    """Parse serialized protocol buffer data in binary form into this message.

    Like :func:`MergeFromString()`, except we clear the object first.

    Raises:
      message.DecodeError if the input cannot be parsed.
    """
    self.Clear()
    return self.MergeFromString(serialized)