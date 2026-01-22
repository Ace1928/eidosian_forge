import time
class LiveBokehChart(object):
    """Callback object that renders a bokeh chart in a jupyter notebook
    that gets updated as the training run proceeds.

    Requires a PandasLogger to collect the data it will render.

    This is an abstract base-class.  Sub-classes define the specific chart.
    """

    def __init__(self, pandas_logger, metric_name, display_freq=10, batch_size=None, frequent=50):
        if pandas_logger:
            self.pandas_logger = pandas_logger
        else:
            self.pandas_logger = PandasLogger(batch_size=batch_size, frequent=frequent)
        self.display_freq = display_freq
        self.last_update = time.time()
        self.metric_name = metric_name
        bokeh.io.output_notebook()
        self.handle = self.setup_chart()

    def setup_chart(self):
        """Render a bokeh object and return a handle to it.
        """
        raise NotImplementedError('Incomplete base class: LiveBokehChart must be sub-classed')

    def update_chart_data(self):
        """Update the bokeh object with new data.
        """
        raise NotImplementedError('Incomplete base class: LiveBokehChart must be sub-classed')

    def interval_elapsed(self):
        """Check whether it is time to update plot.
        Returns
        -------
        Boolean value of whethe to update now
        """
        return time.time() - self.last_update > self.display_freq

    def _push_render(self):
        """Render the plot with bokeh.io and push to notebook.
        """
        bokeh.io.push_notebook(handle=self.handle)
        self.last_update = time.time()

    def _do_update(self):
        """Update the plot chart data and render the updates.
        """
        self.update_chart_data()
        self._push_render()

    def batch_cb(self, param):
        """Callback function after a completed batch.
        """
        if self.interval_elapsed():
            self._do_update()

    def eval_cb(self, param):
        """Callback function after an evaluation.
        """
        self._do_update()

    def callback_args(self):
        """returns **kwargs parameters for model.fit()
        to enable all callbacks.  e.g.
        model.fit(X=train, eval_data=test, **pdlogger.callback_args())
        """
        return {'batch_end_callback': self.batch_cb, 'eval_end_callback': self.eval_cb}