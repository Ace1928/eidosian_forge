from collections import namedtuple
from pyomo.core.expr.numvalue import value as pyo_value
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable, _DynamicDataBase
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.find_nearest_index import (
class IntervalData(_DynamicDataBase):

    def __init__(self, data, intervals, time_set=None, context=None):
        intervals = list(intervals)
        if not intervals == list(sorted(intervals)):
            raise RuntimeError('Intervals are not sorted in increasing order.')
        assert_disjoint_intervals(intervals)
        self._intervals = intervals
        for key, data_list in data.items():
            if len(data_list) != len(intervals):
                raise ValueError('Data lists must have same length as time. Length of time is %s while length of data for key %s is %s.' % (len(intervals), key, len(data_list)))
        super().__init__(data, time_set=time_set, context=context)

    def __eq__(self, other):
        if isinstance(other, IntervalData):
            return self._data == other.get_data() and self._intervals == other.get_intervals()
        else:
            raise TypeError('%s and %s are not comparable' % (self.__class__, other.__class__))

    def get_intervals(self):
        return self._intervals

    def get_data_at_interval_indices(self, indices):
        if _is_iterable(indices):
            index_list = list(sorted(indices))
            interval_list = [self._intervals[i] for i in indices]
            data = {cuid: [values[idx] for idx in index_list] for cuid, values in self._data.items()}
            time_set = self._orig_time_set
            return IntervalData(data, interval_list, time_set=time_set)
        else:
            return ScalarData({cuid: values[indices] for cuid, values in self._data.items()})

    def get_data_at_time(self, time, tolerance=None, prefer_left=True):
        if not _is_iterable(time):
            index = find_nearest_interval_index(self._intervals, time, tolerance=tolerance, prefer_left=prefer_left)
            if index is None:
                raise RuntimeError('Time point %s not found in an interval within tolerance %s' % (time, tolerance))
        else:
            raise RuntimeError('get_data_at_time is not supported with multiple time points for IntervalData. To sample the piecewise-constant data at particular time points, please use interval_to_series from pyomo.contrib.mpc.data.convert')
        return self.get_data_at_interval_indices(index)

    def to_serializable(self):
        """
        Convert to json-serializable object.

        """
        intervals = self._intervals
        data = {str(cuid): [pyo_value(val) for val in values] for cuid, values in self._data.items()}
        return IntervalDataTuple(data, intervals)

    def concatenate(self, other, tolerance=0.0):
        """
        Extend interval list and variable data lists with the intervals
        and variable values in the provided IntervalData

        """
        other_intervals = other.get_intervals()
        intervals = self._intervals
        if len(other_intervals) == 0:
            return
        if other_intervals[0][0] < intervals[-1][1] + tolerance:
            raise ValueError('Initial time point of target, %s, is not greater than final time point of source, %s, within tolerance %s.' % (other_time[0][0], time[-1][1], tolerance))
        self._intervals.extend(other_intervals)
        data = self._data
        other_data = other.get_data()
        for cuid, values in data.items():
            values.extend(other_data[cuid])

    def shift_time_points(self, offset):
        """
        Apply an offset to stored time points.

        """
        self._intervals = [(lo + offset, hi + offset) for lo, hi in self._intervals]

    def extract_variables(self, variables, context=None, copy_values=False):
        """
        Only keep variables specified.

        """
        if copy_values:
            raise NotImplementedError('extract_variables with copy_values=True has not been implemented by %s' % self.__class__)
        data = {}
        if not isinstance(variables, (list, tuple)):
            raise TypeError('extract_values only accepts a list or tuple of variables')
        for var in variables:
            cuid = get_indexed_cuid(var, (self._orig_time_set,), context=context)
            data[cuid] = self._data[cuid]
        return IntervalData(data, self._intervals, time_set=self._orig_time_set)