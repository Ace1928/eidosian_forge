import numpy as np
Return an image with ~`n_points` regularly-spaced nonzero pixels.

    Parameters
    ----------
    ar_shape : tuple of int
        The shape of the desired output image.
    n_points : int
        The desired number of nonzero points.
    dtype : numpy data type, optional
        The desired data type of the output.

    Returns
    -------
    seed_img : array of int or bool
        The desired image.

    Examples
    --------
    >>> regular_seeds((5, 5), 4)
    array([[0, 0, 0, 0, 0],
           [0, 1, 0, 2, 0],
           [0, 0, 0, 0, 0],
           [0, 3, 0, 4, 0],
           [0, 0, 0, 0, 0]])
    