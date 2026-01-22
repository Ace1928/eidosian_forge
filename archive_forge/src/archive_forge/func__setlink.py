import inspect
import warnings
import numpy as np
from scipy import special, stats
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.tools.sm_exceptions import (
from . import links as L, varfuncs as V
def _setlink(self, link):
    """
        Helper method to set the link for a family.

        Raises a ``ValueError`` exception if the link is not available. Note
        that  the error message might not be that informative because it tells
        you that the link should be in the base class for the link function.

        See statsmodels.genmod.generalized_linear_model.GLM for a list of
        appropriate links for each family but note that not all of these are
        currently available.
        """
    self._link = link
    if self._check_link:
        if not isinstance(link, L.Link):
            raise TypeError('The input should be a valid Link object.')
        if hasattr(self, 'links'):
            validlink = max([isinstance(link, _) for _ in self.links])
            if not validlink:
                msg = 'Invalid link for family, should be in %s. (got %s)'
                raise ValueError(msg % (repr(self.links), link))