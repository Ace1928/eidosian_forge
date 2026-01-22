import numpy as np
import pandas as pd
from statsmodels.base.model import LikelihoodModelResults

        Summarize the results of running multiple imputation.

        Parameters
        ----------
        title : str, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            Significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            This holds the summary tables and text, which can be
            printed or converted to various output formats.
        