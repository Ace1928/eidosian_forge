from statsmodels.compat.pandas import deprecate_kwarg
import os
import subprocess
import tempfile
import re
from warnings import warn
import pandas as pd
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (X13NotFoundError,
class X13ArimaAnalysisResult:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def plot(self):
        from statsmodels.graphics.utils import _import_mpl
        plt = _import_mpl()
        fig, axes = plt.subplots(4, 1, sharex=True)
        self.observed.plot(ax=axes[0], legend=False)
        axes[0].set_ylabel('Observed')
        self.seasadj.plot(ax=axes[1], legend=False)
        axes[1].set_ylabel('Seas. Adjusted')
        self.trend.plot(ax=axes[2], legend=False)
        axes[2].set_ylabel('Trend')
        self.irregular.plot(ax=axes[3], legend=False)
        axes[3].set_ylabel('Irregular')
        fig.tight_layout()
        return fig