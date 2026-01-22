import itertools
class MeasurementVariables(VariablesWithIndices):

    def __init__(self):
        """
        This class stores information on which algebraic and differential variables in the Pyomo model are considered measurements.
        """
        super().__init__()
        self.variance = {}

    def set_variable_name_list(self, variable_name_list, variance=1):
        """
        Specify variable names if given strings containing names and indices.

        Parameters
        ----------
        variable_name_list: a ``list`` of ``string``, containing the variable names with indices,
            for e.g. "C['CA', 23, 0]".
        variance: a ``list`` of scalar numbers , which is the variance for this measurement.
        """
        super().set_variable_name_list(variable_name_list)
        if variance is not list:
            variance = [variance] * len(variable_name_list)
        self.variance.update(zip(variable_name_list, variance))

    def add_variables(self, var_name, indices=None, time_index_position=None, variance=1):
        """
        Parameters
        -----------
        var_name: a ``list`` of var names
        indices: a ``dict`` containing indices
            if default (None), no extra indices needed for all var in var_name
            for e.g., {0:["CA", "CB", "CC"], 1: [1,2,3]}.
        time_index_position: an integer indicates which index is the time index
            for e.g., 1 is the time index position in the indices example.
        variance: a scalar number, which is the variance for this measurement.
        """
        added_names = super().add_variables(var_name=var_name, indices=indices, time_index_position=time_index_position)
        if variance is not list:
            variance = [variance] * len(added_names)
        self.variance.update(zip(added_names, variance))

    def check_subset(self, subset_object):
        """
        Check if subset_object is a subset of the current measurement object

        Parameters
        ----------
        subset_object: a measurement object
        """
        for name in subset_object.variable_names:
            if name not in self.variable_names:
                raise ValueError('Measurement not in the set: ', name)
        return True