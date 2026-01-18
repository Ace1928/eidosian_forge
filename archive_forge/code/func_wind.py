def wind():
    """
    Each row represents a level of wind intensity in a cardinal direction, and its frequency.

    Returns:
        A `pandas.DataFrame` with 128 rows and the following columns:
        `['direction', 'strength', 'frequency']`."""
    return _get_dataset('wind')