import logging
from io import BytesIO
from os import PathLike, makedirs, remove
from os.path import exists
import joblib
import numpy as np
from ..utils import Bunch
from ..utils._param_validation import validate_params
from . import get_data_home
from ._base import RemoteFileMetadata, _fetch_remote, _pkl_filepath
Loader for species distribution dataset from Phillips et. al. (2006).

    Read more in the :ref:`User Guide <species_distribution_dataset>`.

    Parameters
    ----------
    data_home : str or path-like, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        coverages : array, shape = [14, 1592, 1212]
            These represent the 14 features measured
            at each point of the map grid.
            The latitude/longitude values for the grid are discussed below.
            Missing data is represented by the value -9999.
        train : record array, shape = (1624,)
            The training points for the data.  Each point has three fields:

            - train['species'] is the species name
            - train['dd long'] is the longitude, in degrees
            - train['dd lat'] is the latitude, in degrees
        test : record array, shape = (620,)
            The test points for the data.  Same format as the training data.
        Nx, Ny : integers
            The number of longitudes (x) and latitudes (y) in the grid
        x_left_lower_corner, y_left_lower_corner : floats
            The (x,y) position of the lower-left corner, in degrees
        grid_size : float
            The spacing between points of the grid, in degrees

    Notes
    -----

    This dataset represents the geographic distribution of species.
    The dataset is provided by Phillips et. al. (2006).

    The two species are:

    - `"Bradypus variegatus"
      <http://www.iucnredlist.org/details/3038/0>`_ ,
      the Brown-throated Sloth.

    - `"Microryzomys minutus"
      <http://www.iucnredlist.org/details/13408/0>`_ ,
      also known as the Forest Small Rice Rat, a rodent that lives in Peru,
      Colombia, Ecuador, Peru, and Venezuela.

    - For an example of using this dataset with scikit-learn, see
      :ref:`examples/applications/plot_species_distribution_modeling.py
      <sphx_glr_auto_examples_applications_plot_species_distribution_modeling.py>`.

    References
    ----------

    * `"Maximum entropy modeling of species geographic distributions"
      <http://rob.schapire.net/papers/ecolmod.pdf>`_
      S. J. Phillips, R. P. Anderson, R. E. Schapire - Ecological Modelling,
      190:231-259, 2006.

    Examples
    --------
    >>> from sklearn.datasets import fetch_species_distributions
    >>> species = fetch_species_distributions()
    >>> species.train[:5]
    array([(b'microryzomys_minutus', -64.7   , -17.85  ),
           (b'microryzomys_minutus', -67.8333, -16.3333),
           (b'microryzomys_minutus', -67.8833, -16.3   ),
           (b'microryzomys_minutus', -67.8   , -16.2667),
           (b'microryzomys_minutus', -67.9833, -15.9   )],
          dtype=[('species', 'S22'), ('dd long', '<f4'), ('dd lat', '<f4')])
    