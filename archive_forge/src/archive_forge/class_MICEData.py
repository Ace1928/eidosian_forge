import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
class MICEData:
    __doc__ = "    Wrap a data set to allow missing data handling with MICE.\n\n    Parameters\n    ----------\n    data : Pandas data frame\n        The data set, which is copied internally.\n    perturbation_method : str\n        The default perturbation method\n    k_pmm : int\n        The number of nearest neighbors to use during predictive mean\n        matching.  Can also be specified in `fit`.\n    history_callback : function\n        A function that is called after each complete imputation\n        cycle.  The return value is appended to `history`.  The\n        MICEData object is passed as the sole argument to\n        `history_callback`.\n\n    Notes\n    -----\n    Allowed perturbation methods are 'gaussian' (the model parameters\n    are set to a draw from the Gaussian approximation to the posterior\n    distribution), and 'boot' (the model parameters are set to the\n    estimated values obtained when fitting a bootstrapped version of\n    the data set).\n\n    `history_callback` can be implemented to have side effects such as\n    saving the current imputed data set to disk.\n\n    Examples\n    --------\n    Draw 20 imputations from a data set called `data` and save them in\n    separate files with filename pattern `dataXX.csv`.  The variables\n    other than `x1` are imputed using linear models fit with OLS, with\n    mean structures containing main effects of all other variables in\n    `data`.  The variable named `x1` has a conditional mean structure\n    that includes an additional term for x2^2.\n    {_mice_data_example_1}\n    ".format(_mice_data_example_1=_mice_data_example_1)

    def __init__(self, data, perturbation_method='gaussian', k_pmm=20, history_callback=None):
        if data.columns.dtype != np.dtype('O'):
            msg = 'MICEData data column names should be string type'
            raise ValueError(msg)
        self.regularized = dict()
        self.data = data.dropna(how='all').reset_index(drop=True)
        self.history_callback = history_callback
        self.history = []
        self.predict_kwds = {}
        self.perturbation_method = defaultdict(lambda: perturbation_method)
        self.ix_obs = {}
        self.ix_miss = {}
        for col in self.data.columns:
            ix_obs, ix_miss = self._split_indices(self.data[col])
            self.ix_obs[col] = ix_obs
            self.ix_miss[col] = ix_miss
        self.models = {}
        self.results = {}
        self.conditional_formula = {}
        self.init_kwds = defaultdict(dict)
        self.fit_kwds = defaultdict(dict)
        self.model_class = {}
        self.params = {}
        for vname in data.columns:
            self.set_imputer(vname)
        vnames = list(data.columns)
        nmiss = [len(self.ix_miss[v]) for v in vnames]
        nmiss = np.asarray(nmiss)
        ii = np.argsort(nmiss)
        ii = ii[sum(nmiss == 0):]
        self._cycle_order = [vnames[i] for i in ii]
        self._initial_imputation()
        self.k_pmm = k_pmm

    def next_sample(self):
        """
        Returns the next imputed dataset in the imputation process.

        Returns
        -------
        data : array_like
            An imputed dataset from the MICE chain.

        Notes
        -----
        `MICEData` does not have a `skip` parameter.  Consecutive
        values returned by `next_sample` are immediately consecutive
        in the imputation chain.

        The returned value is a reference to the data attribute of
        the class and should be copied before making any changes.
        """
        self.update_all(1)
        return self.data

    def _initial_imputation(self):
        """
        Use a PMM-like procedure for initial imputed values.

        For each variable, missing values are imputed as the observed
        value that is closest to the mean over all observed values.
        """
        imp_values = {}
        for col in self.data.columns:
            di = self.data[col] - self.data[col].mean()
            di = np.abs(di)
            ix = di.idxmin()
            imp_values[col] = self.data[col].loc[ix]
        self.data.fillna(imp_values, inplace=True)

    def _split_indices(self, vec):
        null = pd.isnull(vec)
        ix_obs = np.flatnonzero(~null)
        ix_miss = np.flatnonzero(null)
        if len(ix_obs) == 0:
            raise ValueError('variable to be imputed has no observed values')
        return (ix_obs, ix_miss)

    def set_imputer(self, endog_name, formula=None, model_class=None, init_kwds=None, fit_kwds=None, predict_kwds=None, k_pmm=20, perturbation_method=None, regularized=False):
        """
        Specify the imputation process for a single variable.

        Parameters
        ----------
        endog_name : str
            Name of the variable to be imputed.
        formula : str
            Conditional formula for imputation. Defaults to a formula
            with main effects for all other variables in dataset.  The
            formula should only include an expression for the mean
            structure, e.g. use 'x1 + x2' not 'x4 ~ x1 + x2'.
        model_class : statsmodels model
            Conditional model for imputation. Defaults to OLS.  See below
            for more information.
        init_kwds : dit-like
            Keyword arguments passed to the model init method.
        fit_kwds : dict-like
            Keyword arguments passed to the model fit method.
        predict_kwds : dict-like
            Keyword arguments passed to the model predict method.
        k_pmm : int
            Determines number of neighboring observations from which
            to randomly sample when using predictive mean matching.
        perturbation_method : str
            Either 'gaussian' or 'bootstrap'. Determines the method
            for perturbing parameters in the imputation model.  If
            None, uses the default specified at class initialization.
        regularized : dict
            If regularized[name]=True, `fit_regularized` rather than
            `fit` is called when fitting imputation models for this
            variable.  When regularized[name]=True for any variable,
            perturbation_method must be set to boot.

        Notes
        -----
        The model class must meet the following conditions:
            * A model must have a 'fit' method that returns an object.
            * The object returned from `fit` must have a `params` attribute
              that is an array-like object.
            * The object returned from `fit` must have a cov_params method
              that returns a square array-like object.
            * The model must have a `predict` method.
        """
        if formula is None:
            main_effects = [x for x in self.data.columns if x != endog_name]
            fml = endog_name + ' ~ ' + ' + '.join(main_effects)
            self.conditional_formula[endog_name] = fml
        else:
            fml = endog_name + ' ~ ' + formula
            self.conditional_formula[endog_name] = fml
        if model_class is None:
            self.model_class[endog_name] = OLS
        else:
            self.model_class[endog_name] = model_class
        if init_kwds is not None:
            self.init_kwds[endog_name] = init_kwds
        if fit_kwds is not None:
            self.fit_kwds[endog_name] = fit_kwds
        if predict_kwds is not None:
            self.predict_kwds[endog_name] = predict_kwds
        if perturbation_method is not None:
            self.perturbation_method[endog_name] = perturbation_method
        self.k_pmm = k_pmm
        self.regularized[endog_name] = regularized

    def _store_changes(self, col, vals):
        """
        Fill in dataset with imputed values.

        Parameters
        ----------
        col : str
            Name of variable to be filled in.
        vals : ndarray
            Array of imputed values to use for filling-in missing values.
        """
        ix = self.ix_miss[col]
        if len(ix) > 0:
            self.data.iloc[ix, self.data.columns.get_loc(col)] = np.atleast_1d(vals)

    def update_all(self, n_iter=1):
        """
        Perform a specified number of MICE iterations.

        Parameters
        ----------
        n_iter : int
            The number of updates to perform.  Only the result of the
            final update will be available.

        Notes
        -----
        The imputed values are stored in the class attribute `self.data`.
        """
        for k in range(n_iter):
            for vname in self._cycle_order:
                self.update(vname)
        if self.history_callback is not None:
            hv = self.history_callback(self)
            self.history.append(hv)

    def get_split_data(self, vname):
        """
        Return endog and exog for imputation of a given variable.

        Parameters
        ----------
        vname : str
           The variable for which the split data is returned.

        Returns
        -------
        endog_obs : DataFrame
            Observed values of the variable to be imputed.
        exog_obs : DataFrame
            Current values of the predictors where the variable to be
            imputed is observed.
        exog_miss : DataFrame
            Current values of the predictors where the variable to be
            Imputed is missing.
        init_kwds : dict-like
            The init keyword arguments for `vname`, processed through Patsy
            as required.
        fit_kwds : dict-like
            The fit keyword arguments for `vname`, processed through Patsy
            as required.
        """
        formula = self.conditional_formula[vname]
        endog, exog = patsy.dmatrices(formula, self.data, return_type='dataframe')
        ixo = self.ix_obs[vname]
        endog_obs = np.require(endog.iloc[ixo], requirements='W')
        exog_obs = np.require(exog.iloc[ixo, :], requirements='W')
        ixm = self.ix_miss[vname]
        exog_miss = np.require(exog.iloc[ixm, :], requirements='W')
        predict_obs_kwds = {}
        if vname in self.predict_kwds:
            kwds = self.predict_kwds[vname]
            predict_obs_kwds = self._process_kwds(kwds, ixo)
        predict_miss_kwds = {}
        if vname in self.predict_kwds:
            kwds = self.predict_kwds[vname]
            predict_miss_kwds = self._process_kwds(kwds, ixo)
        return (endog_obs, exog_obs, exog_miss, predict_obs_kwds, predict_miss_kwds)

    def _process_kwds(self, kwds, ix):
        kwds = kwds.copy()
        for k in kwds:
            v = kwds[k]
            if isinstance(v, PatsyFormula):
                mat = patsy.dmatrix(v.formula, self.data, return_type='dataframe')
                mat = np.require(mat, requirements='W')[ix, :]
                if mat.shape[1] == 1:
                    mat = mat[:, 0]
                kwds[k] = mat
        return kwds

    def get_fitting_data(self, vname):
        """
        Return the data needed to fit a model for imputation.

        The data is used to impute variable `vname`, and therefore
        only includes cases for which `vname` is observed.

        Values of type `PatsyFormula` in `init_kwds` or `fit_kwds` are
        processed through Patsy and subset to align with the model's
        endog and exog.

        Parameters
        ----------
        vname : str
           The variable for which the fitting data is returned.

        Returns
        -------
        endog : DataFrame
            Observed values of `vname`.
        exog : DataFrame
            Regression design matrix for imputing `vname`.
        init_kwds : dict-like
            The init keyword arguments for `vname`, processed through Patsy
            as required.
        fit_kwds : dict-like
            The fit keyword arguments for `vname`, processed through Patsy
            as required.
        """
        ix = self.ix_obs[vname]
        formula = self.conditional_formula[vname]
        endog, exog = patsy.dmatrices(formula, self.data, return_type='dataframe')
        endog = np.require(endog.iloc[ix, 0], requirements='W')
        exog = np.require(exog.iloc[ix, :], requirements='W')
        init_kwds = self._process_kwds(self.init_kwds[vname], ix)
        fit_kwds = self._process_kwds(self.fit_kwds[vname], ix)
        return (endog, exog, init_kwds, fit_kwds)

    def plot_missing_pattern(self, ax=None, row_order='pattern', column_order='pattern', hide_complete_rows=False, hide_complete_columns=False, color_row_patterns=True):
        """
        Generate an image showing the missing data pattern.

        Parameters
        ----------
        ax : AxesSubplot
            Axes on which to draw the plot.
        row_order : str
            The method for ordering the rows.  Must be one of 'pattern',
            'proportion', or 'raw'.
        column_order : str
            The method for ordering the columns.  Must be one of 'pattern',
            'proportion', or 'raw'.
        hide_complete_rows : bool
            If True, rows with no missing values are not drawn.
        hide_complete_columns : bool
            If True, columns with no missing values are not drawn.
        color_row_patterns : bool
            If True, color the unique row patterns, otherwise use grey
            and white as colors.

        Returns
        -------
        A figure containing a plot of the missing data pattern.
        """
        miss = np.zeros(self.data.shape)
        cols = self.data.columns
        for j, col in enumerate(cols):
            ix = self.ix_miss[col]
            miss[ix, j] = 1
        if column_order == 'proportion':
            ix = np.argsort(miss.mean(0))
        elif column_order == 'pattern':
            cv = np.cov(miss.T)
            u, s, vt = np.linalg.svd(cv, 0)
            ix = np.argsort(cv[:, 0])
        elif column_order == 'raw':
            ix = np.arange(len(cols))
        else:
            raise ValueError(column_order + ' is not an allowed value for `column_order`.')
        miss = miss[:, ix]
        cols = [cols[i] for i in ix]
        if row_order == 'proportion':
            ix = np.argsort(miss.mean(1))
        elif row_order == 'pattern':
            x = 2 ** np.arange(miss.shape[1])
            rky = np.dot(miss, x)
            ix = np.argsort(rky)
        elif row_order == 'raw':
            ix = np.arange(miss.shape[0])
        else:
            raise ValueError(row_order + ' is not an allowed value for `row_order`.')
        miss = miss[ix, :]
        if hide_complete_rows:
            ix = np.flatnonzero((miss == 1).any(1))
            miss = miss[ix, :]
        if hide_complete_columns:
            ix = np.flatnonzero((miss == 1).any(0))
            miss = miss[:, ix]
            cols = [cols[i] for i in ix]
        from statsmodels.graphics import utils as gutils
        from matplotlib.colors import LinearSegmentedColormap
        if ax is None:
            fig, ax = gutils.create_mpl_ax(ax)
        else:
            fig = ax.get_figure()
        if color_row_patterns:
            x = 2 ** np.arange(miss.shape[1])
            rky = np.dot(miss, x)
            _, rcol = np.unique(rky, return_inverse=True)
            miss *= 1 + rcol[:, None]
            ax.imshow(miss, aspect='auto', interpolation='nearest', cmap='gist_ncar_r')
        else:
            cmap = LinearSegmentedColormap.from_list('_', ['white', 'darkgrey'])
            ax.imshow(miss, aspect='auto', interpolation='nearest', cmap=cmap)
        ax.set_ylabel('Cases')
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90)
        return fig

    def plot_bivariate(self, col1_name, col2_name, lowess_args=None, lowess_min_n=40, jitter=None, plot_points=True, ax=None):
        """
        Plot observed and imputed values for two variables.

        Displays a scatterplot of one variable against another.  The
        points are colored according to whether the values are
        observed or imputed.

        Parameters
        ----------
        col1_name : str
            The variable to be plotted on the horizontal axis.
        col2_name : str
            The variable to be plotted on the vertical axis.
        lowess_args : dictionary
            A dictionary of dictionaries, keys are 'ii', 'io', 'oi'
            and 'oo', where 'o' denotes 'observed' and 'i' denotes
            imputed.  See Notes for details.
        lowess_min_n : int
            Minimum sample size to plot a lowess fit
        jitter : float or tuple
            Standard deviation for jittering points in the plot.
            Either a single scalar applied to both axes, or a tuple
            containing x-axis jitter and y-axis jitter, respectively.
        plot_points : bool
            If True, the data points are plotted.
        ax : AxesSubplot
            Axes on which to plot, created if not provided.

        Returns
        -------
        The matplotlib figure on which the plot id drawn.
        """
        from statsmodels.graphics import utils as gutils
        from statsmodels.nonparametric.smoothers_lowess import lowess
        if lowess_args is None:
            lowess_args = {}
        if ax is None:
            fig, ax = gutils.create_mpl_ax(ax)
        else:
            fig = ax.get_figure()
        ax.set_position([0.1, 0.1, 0.7, 0.8])
        ix1i = self.ix_miss[col1_name]
        ix1o = self.ix_obs[col1_name]
        ix2i = self.ix_miss[col2_name]
        ix2o = self.ix_obs[col2_name]
        ix_ii = np.intersect1d(ix1i, ix2i)
        ix_io = np.intersect1d(ix1i, ix2o)
        ix_oi = np.intersect1d(ix1o, ix2i)
        ix_oo = np.intersect1d(ix1o, ix2o)
        vec1 = np.require(self.data[col1_name], requirements='W')
        vec2 = np.require(self.data[col2_name], requirements='W')
        if jitter is not None:
            if np.isscalar(jitter):
                jitter = (jitter, jitter)
            vec1 += jitter[0] * np.random.normal(size=len(vec1))
            vec2 += jitter[1] * np.random.normal(size=len(vec2))
        keys = ['oo', 'io', 'oi', 'ii']
        lak = {'i': 'imp', 'o': 'obs'}
        ixs = {'ii': ix_ii, 'io': ix_io, 'oi': ix_oi, 'oo': ix_oo}
        color = {'oo': 'grey', 'ii': 'red', 'io': 'orange', 'oi': 'lime'}
        if plot_points:
            for ky in keys:
                ix = ixs[ky]
                lab = lak[ky[0]] + '/' + lak[ky[1]]
                ax.plot(vec1[ix], vec2[ix], 'o', color=color[ky], label=lab, alpha=0.6)
        for ky in keys:
            ix = ixs[ky]
            if len(ix) < lowess_min_n:
                continue
            if ky in lowess_args:
                la = lowess_args[ky]
            else:
                la = {}
            ix = ixs[ky]
            lfit = lowess(vec2[ix], vec1[ix], **la)
            if plot_points:
                ax.plot(lfit[:, 0], lfit[:, 1], '-', color=color[ky], alpha=0.6, lw=4)
            else:
                lab = lak[ky[0]] + '/' + lak[ky[1]]
                ax.plot(lfit[:, 0], lfit[:, 1], '-', color=color[ky], alpha=0.6, lw=4, label=lab)
        ha, la = ax.get_legend_handles_labels()
        pad = 0.0001 if plot_points else 0.5
        leg = fig.legend(ha, la, loc='center right', numpoints=1, handletextpad=pad)
        leg.draw_frame(False)
        ax.set_xlabel(col1_name)
        ax.set_ylabel(col2_name)
        return fig

    def plot_fit_obs(self, col_name, lowess_args=None, lowess_min_n=40, jitter=None, plot_points=True, ax=None):
        """
        Plot fitted versus imputed or observed values as a scatterplot.

        Parameters
        ----------
        col_name : str
            The variable to be plotted on the horizontal axis.
        lowess_args : dict-like
            Keyword arguments passed to lowess fit.  A dictionary of
            dictionaries, keys are 'o' and 'i' denoting 'observed' and
            'imputed', respectively.
        lowess_min_n : int
            Minimum sample size to plot a lowess fit
        jitter : float or tuple
            Standard deviation for jittering points in the plot.
            Either a single scalar applied to both axes, or a tuple
            containing x-axis jitter and y-axis jitter, respectively.
        plot_points : bool
            If True, the data points are plotted.
        ax : AxesSubplot
            Axes on which to plot, created if not provided.

        Returns
        -------
        The matplotlib figure on which the plot is drawn.
        """
        from statsmodels.graphics import utils as gutils
        from statsmodels.nonparametric.smoothers_lowess import lowess
        if lowess_args is None:
            lowess_args = {}
        if ax is None:
            fig, ax = gutils.create_mpl_ax(ax)
        else:
            fig = ax.get_figure()
        ax.set_position([0.1, 0.1, 0.7, 0.8])
        ixi = self.ix_miss[col_name]
        ixo = self.ix_obs[col_name]
        vec1 = np.require(self.data[col_name], requirements='W')
        formula = self.conditional_formula[col_name]
        endog, exog = patsy.dmatrices(formula, self.data, return_type='dataframe')
        results = self.results[col_name]
        vec2 = results.predict(exog=exog)
        vec2 = self._get_predicted(vec2)
        if jitter is not None:
            if np.isscalar(jitter):
                jitter = (jitter, jitter)
            vec1 += jitter[0] * np.random.normal(size=len(vec1))
            vec2 += jitter[1] * np.random.normal(size=len(vec2))
        keys = ['o', 'i']
        ixs = {'o': ixo, 'i': ixi}
        lak = {'o': 'obs', 'i': 'imp'}
        color = {'o': 'orange', 'i': 'lime'}
        if plot_points:
            for ky in keys:
                ix = ixs[ky]
                ax.plot(vec1[ix], vec2[ix], 'o', color=color[ky], label=lak[ky], alpha=0.6)
        for ky in keys:
            ix = ixs[ky]
            if len(ix) < lowess_min_n:
                continue
            if ky in lowess_args:
                la = lowess_args[ky]
            else:
                la = {}
            ix = ixs[ky]
            lfit = lowess(vec2[ix], vec1[ix], **la)
            ax.plot(lfit[:, 0], lfit[:, 1], '-', color=color[ky], alpha=0.6, lw=4, label=lak[ky])
        ha, la = ax.get_legend_handles_labels()
        leg = fig.legend(ha, la, loc='center right', numpoints=1)
        leg.draw_frame(False)
        ax.set_xlabel(col_name + ' observed or imputed')
        ax.set_ylabel(col_name + ' fitted')
        return fig

    def plot_imputed_hist(self, col_name, ax=None, imp_hist_args=None, obs_hist_args=None, all_hist_args=None):
        """
        Display imputed values for one variable as a histogram.

        Parameters
        ----------
        col_name : str
            The name of the variable to be plotted.
        ax : AxesSubplot
            An axes on which to draw the histograms.  If not provided,
            one is created.
        imp_hist_args : dict
            Keyword arguments to be passed to pyplot.hist when
            creating the histogram for imputed values.
        obs_hist_args : dict
            Keyword arguments to be passed to pyplot.hist when
            creating the histogram for observed values.
        all_hist_args : dict
            Keyword arguments to be passed to pyplot.hist when
            creating the histogram for all values.

        Returns
        -------
        The matplotlib figure on which the histograms were drawn
        """
        from statsmodels.graphics import utils as gutils
        if imp_hist_args is None:
            imp_hist_args = {}
        if obs_hist_args is None:
            obs_hist_args = {}
        if all_hist_args is None:
            all_hist_args = {}
        if ax is None:
            fig, ax = gutils.create_mpl_ax(ax)
        else:
            fig = ax.get_figure()
        ax.set_position([0.1, 0.1, 0.7, 0.8])
        ixm = self.ix_miss[col_name]
        ixo = self.ix_obs[col_name]
        imp = self.data[col_name].iloc[ixm]
        obs = self.data[col_name].iloc[ixo]
        for di in (imp_hist_args, obs_hist_args, all_hist_args):
            if 'histtype' not in di:
                di['histtype'] = 'step'
        ha, la = ([], [])
        if len(imp) > 0:
            h = ax.hist(np.asarray(imp), **imp_hist_args)
            ha.append(h[-1][0])
            la.append('Imp')
        h1 = ax.hist(np.asarray(obs), **obs_hist_args)
        h2 = ax.hist(np.asarray(self.data[col_name]), **all_hist_args)
        ha.extend([h1[-1][0], h2[-1][0]])
        la.extend(['Obs', 'All'])
        leg = fig.legend(ha, la, loc='center right', numpoints=1)
        leg.draw_frame(False)
        ax.set_xlabel(col_name)
        ax.set_ylabel('Frequency')
        return fig

    def _boot_kwds(self, kwds, rix):
        for k in kwds:
            v = kwds[k]
            if not isinstance(v, np.ndarray):
                continue
            if v.ndim == 1 and v.shape[0] == len(rix):
                kwds[k] = v[rix]
            if v.ndim == 2 and v.shape[0] == len(rix):
                kwds[k] = v[rix, :]
        return kwds

    def _perturb_bootstrap(self, vname):
        """
        Perturbs the model's parameters using a bootstrap.
        """
        endog, exog, init_kwds, fit_kwds = self.get_fitting_data(vname)
        m = len(endog)
        rix = np.random.randint(0, m, m)
        endog = endog[rix]
        exog = exog[rix, :]
        init_kwds = self._boot_kwds(init_kwds, rix)
        fit_kwds = self._boot_kwds(fit_kwds, rix)
        klass = self.model_class[vname]
        self.models[vname] = klass(endog, exog, **init_kwds)
        if vname in self.regularized and self.regularized[vname]:
            self.results[vname] = self.models[vname].fit_regularized(**fit_kwds)
        else:
            self.results[vname] = self.models[vname].fit(**fit_kwds)
        self.params[vname] = self.results[vname].params

    def _perturb_gaussian(self, vname):
        """
        Gaussian perturbation of model parameters.

        The normal approximation to the sampling distribution of the
        parameter estimates is used to define the mean and covariance
        structure of the perturbation distribution.
        """
        endog, exog, init_kwds, fit_kwds = self.get_fitting_data(vname)
        klass = self.model_class[vname]
        self.models[vname] = klass(endog, exog, **init_kwds)
        self.results[vname] = self.models[vname].fit(**fit_kwds)
        cov = self.results[vname].cov_params()
        mu = self.results[vname].params
        self.params[vname] = np.random.multivariate_normal(mean=mu, cov=cov)

    def perturb_params(self, vname):
        if self.perturbation_method[vname] == 'gaussian':
            self._perturb_gaussian(vname)
        elif self.perturbation_method[vname] == 'boot':
            self._perturb_bootstrap(vname)
        else:
            raise ValueError('unknown perturbation method')

    def impute(self, vname):
        self.impute_pmm(vname)

    def update(self, vname):
        """
        Impute missing values for a single variable.

        This is a two-step process in which first the parameters are
        perturbed, then the missing values are re-imputed.

        Parameters
        ----------
        vname : str
            The name of the variable to be updated.
        """
        self.perturb_params(vname)
        self.impute(vname)

    def _get_predicted(self, obj):
        if isinstance(obj, np.ndarray):
            return obj
        elif isinstance(obj, pd.Series):
            return obj.values
        elif hasattr(obj, 'predicted_values'):
            return obj.predicted_values
        else:
            raise ValueError('cannot obtain predicted values from %s' % obj.__class__)

    def impute_pmm(self, vname):
        """
        Use predictive mean matching to impute missing values.

        Notes
        -----
        The `perturb_params` method must be called first to define the
        model.
        """
        k_pmm = self.k_pmm
        endog_obs, exog_obs, exog_miss, predict_obs_kwds, predict_miss_kwds = self.get_split_data(vname)
        model = self.models[vname]
        pendog_obs = model.predict(self.params[vname], exog_obs, **predict_obs_kwds)
        pendog_miss = model.predict(self.params[vname], exog_miss, **predict_miss_kwds)
        pendog_obs = self._get_predicted(pendog_obs)
        pendog_miss = self._get_predicted(pendog_miss)
        ii = np.argsort(pendog_obs)
        endog_obs = endog_obs[ii]
        pendog_obs = pendog_obs[ii]
        ix = np.searchsorted(pendog_obs, pendog_miss)
        ixm = ix[:, None] + np.arange(-k_pmm, k_pmm)[None, :]
        msk = np.nonzero((ixm < 0) | (ixm > len(endog_obs) - 1))
        ixm = np.clip(ixm, 0, len(endog_obs) - 1)
        dx = pendog_miss[:, None] - pendog_obs[ixm]
        dx = np.abs(dx)
        dx[msk] = np.inf
        dxi = np.argsort(dx, 1)[:, 0:k_pmm]
        ir = np.random.randint(0, k_pmm, len(pendog_miss))
        jj = np.arange(dxi.shape[0])
        ix = dxi[jj, ir]
        iz = ixm[jj, ix]
        imputed_miss = np.array(endog_obs[iz]).squeeze()
        self._store_changes(vname, imputed_miss)