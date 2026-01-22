import pandas as pd
from .utils import series_to_line
class _SimpleStreamingFileReader(_BaseReader):

    def __init__(self, file_name, sep, has_header, group_feature_num=None):
        super(_SimpleStreamingFileReader, self).__init__(sep, group_feature_num)
        self._has_header = has_header
        self._file_name = file_name

    def lines_generator(self):
        with open(self._file_name, 'r') as file:
            if self._has_header:
                file.readline()
            for num, line in enumerate(file):
                if self._group_feature_num is None:
                    group_id = num
                else:
                    features = line.strip().split(self._sep, self._group_feature_num + 1)
                    group_id = features[self._group_feature_num]
                yield (int(float(group_id)), line)