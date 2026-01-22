import pandas as pd
from .utils import series_to_line
class _SimpleDataReader(_BaseReader):

    def __init__(self, data, sep, group_feature_num=None):
        super(_SimpleDataReader, self).__init__(sep, group_feature_num)
        self._data = pd.DataFrame(data)

    def lines_generator(self):
        for num, (index, line) in enumerate(self._data.iterrows()):
            if self._group_feature_num is None:
                yield (num, series_to_line(line, self._sep) + '\n')
            else:
                yield (line.iloc[self._group_feature_num], series_to_line(line, self._sep) + '\n')

    def get_matrix(self):
        return self._data