import itertools
class VariablesWithIndices:

    def __init__(self):
        """This class provides utility methods for DesignVariables and MeasurementVariables to create
        lists of Pyomo variable names with an arbitrary number of indices.
        """
        self.variable_names = []
        self.variable_names_value = {}
        self.lower_bounds = {}
        self.upper_bounds = {}

    def set_variable_name_list(self, variable_name_list):
        """
        Specify variable names with its full name.

        Parameters
        ----------
        variable_name_list: a ``list`` of ``string``, containing the variable names with indices,
            for e.g. "C['CA', 23, 0]".
        """
        self.variable_names.extend(variable_name_list)

    def add_variables(self, var_name, indices=None, time_index_position=None, values=None, lower_bounds=None, upper_bounds=None):
        """
        Used for generating string names with indices.

        Parameters
        ----------
        var_name: variable name in ``string``
        indices: a ``dict`` containing indices
            if default (None), no extra indices needed for all var in var_name
            for e.g., {0:["CA", "CB", "CC"], 1: [1,2,3]}.
        time_index_position: an integer indicates which index is the time index
            for e.g., 1 is the time index position in the indices example.
        values: a ``list`` containing values which has the same shape of flattened variables
            default choice is None, means there is no give nvalues
        lower_bounds: a ``list `` of lower bounds. If given a scalar number, it is set as the lower bounds for all variables.
        upper_bounds: a ``list`` of upper bounds. If given a scalar number, it is set as the upper bounds for all variables.

        Returns
        -------
        if not defining values, return a set of variable names
        if defining values, return a dictionary of variable names and its value
        """
        added_names = self._generate_variable_names_with_indices(var_name, indices=indices, time_index_position=time_index_position)
        self._check_valid_input(len(added_names), var_name, indices, time_index_position, values, lower_bounds, upper_bounds)
        if values:
            self.variable_names_value.update(zip(added_names, values))
        if lower_bounds:
            if type(lower_bounds) in [int, float]:
                lower_bounds = [lower_bounds] * len(added_names)
            self.lower_bounds.update(zip(added_names, lower_bounds))
        if upper_bounds:
            if type(upper_bounds) in [int, float]:
                upper_bounds = [upper_bounds] * len(added_names)
            self.upper_bounds.update(zip(added_names, upper_bounds))
        return added_names

    def _generate_variable_names_with_indices(self, var_name, indices=None, time_index_position=None):
        """
        Used for generating string names with indices.

        Parameters
        ----------
        var_name: a ``list`` of var names
        indices: a ``dict`` containing indices
            if default (None), no extra indices needed for all var in var_name
            for e.g., {0:["CA", "CB", "CC"], 1: [1,2,3]}.
        time_index_position: an integer indicates which index is the time index
            for e.g., 1 is the time index position in the indices example.
        """
        all_index_list = []
        if indices:
            for index_pointer in indices:
                all_index_list.append(indices[index_pointer])
        all_variable_indices = list(itertools.product(*all_index_list))
        added_names = []
        for index_instance in all_variable_indices:
            var_name_index_string = var_name + '['
            for i, idx in enumerate(index_instance):
                var_name_index_string += str(idx)
                if i == len(index_instance) - 1:
                    var_name_index_string += ']'
                else:
                    var_name_index_string += ','
            self.variable_names.append(var_name_index_string)
            added_names.append(var_name_index_string)
        return added_names

    def _check_valid_input(self, len_indices, var_name, indices, time_index_position, values, lower_bounds, upper_bounds):
        """
        Check if the measurement information provided are valid to use.
        """
        assert type(var_name) is str, 'var_name should be a string.'
        if time_index_position not in indices:
            raise ValueError('time index cannot be found in indices.')
        if values and len(values) != len_indices:
            raise ValueError('Values is of different length with indices.')
        if lower_bounds and type(lower_bounds) == list and (len(lower_bounds) != len_indices):
            raise ValueError('Lowerbounds is of different length with indices.')
        if upper_bounds and type(upper_bounds) == list and (len(upper_bounds) != len_indices):
            raise ValueError('Upperbounds is of different length with indices.')