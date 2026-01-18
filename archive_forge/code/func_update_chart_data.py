import time
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