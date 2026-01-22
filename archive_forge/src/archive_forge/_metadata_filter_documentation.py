 Callable to be used with FilteredTraitObserver for filtering traits
    with the given metadata name.

    This filter returns true if the metadata value is not None, false
    if the metadata is not defined or the value is None.

    Attributes
    ----------
    metadata_name : str
        Name of the metadata to filter traits with.
    