
    Split input data to train and test data.

    Parameters
    ----------
    df : modin.pandas.DataFrame / modin.pandas.Series
        Data to split.
    **options : dict
        Keyword arguments. If `train_size` key isn't provided
        `train_size` will be 0.75.

    Returns
    -------
    tuple
        A pair of modin.pandas.DataFrame / modin.pandas.Series.
    