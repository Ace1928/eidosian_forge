import time
class LiveLearningCurve(LiveBokehChart):
    """Draws a learning curve with training & validation metrics
    over time as the network trains.
    """

    def __init__(self, metric_name, display_freq=10, frequent=50):
        self.frequent = frequent
        self.start_time = datetime.datetime.now()
        self._data = {'train': {'elapsed': []}, 'eval': {'elapsed': []}}
        super(LiveLearningCurve, self).__init__(None, metric_name, display_freq, frequent)

    def setup_chart(self):
        self.fig = bokeh.plotting.Figure(x_axis_type='datetime', x_axis_label='Training time')
        self.x_axis_val1 = []
        self.y_axis_val1 = []
        self.train1 = self.fig.line(self.x_axis_val1, self.y_axis_val1, line_dash='dotted', alpha=0.3, legend='train')
        self.train2 = self.fig.circle(self.x_axis_val1, self.y_axis_val1, size=1.5, line_alpha=0.3, fill_alpha=0.3, legend='train')
        self.train2.visible = False
        self.x_axis_val2 = []
        self.y_axis_val2 = []
        self.valid1 = self.fig.line(self.x_axis_val2, self.y_axis_val2, line_color='green', line_width=2, legend='validation')
        self.valid2 = self.fig.circle(self.x_axis_val2, self.y_axis_val2, line_color='green', line_width=2, legend=None)
        self.fig.legend.location = 'bottom_right'
        self.fig.yaxis.axis_label = self.metric_name
        return bokeh.plotting.show(self.fig, notebook_handle=True)

    def _do_update(self):
        self.update_chart_data()
        self._push_render()

    def batch_cb(self, param):
        if param.nbatch % self.frequent == 0:
            self._process_batch(param, 'train')
        if self.interval_elapsed():
            self._do_update()

    def eval_cb(self, param):
        self._process_batch(param, 'eval')
        self._do_update()

    def _process_batch(self, param, df_name):
        """Update selected dataframe after a completed batch
        Parameters
        ----------
        df_name : str
            Selected dataframe name needs to be modified.
        """
        if param.eval_metric is not None:
            metrics = dict(param.eval_metric.get_name_value())
            param.eval_metric.reset()
        else:
            metrics = {}
        metrics['elapsed'] = datetime.datetime.now() - self.start_time
        for key, value in metrics.items():
            if key not in self._data[df_name]:
                self._data[df_name][key] = []
            self._data[df_name][key].append(value)

    def update_chart_data(self):
        dataframe = self._data['train']
        if len(dataframe['elapsed']):
            _extend(self.x_axis_val1, dataframe['elapsed'])
            _extend(self.y_axis_val1, dataframe[self.metric_name])
        dataframe = self._data['eval']
        if len(dataframe['elapsed']):
            _extend(self.x_axis_val2, dataframe['elapsed'])
            _extend(self.y_axis_val2, dataframe[self.metric_name])
        if len(dataframe) > 10:
            self.train1.visible = False
            self.train2.visible = True