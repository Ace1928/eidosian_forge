class FilterElement:

    def __init__(self, objective, feasible):
        self.objective = objective
        self.feasible = feasible

    def compare(self, filterElement):
        """
        This method compares the objective and feasibility values
        of the filter element to determine whether or not the filter element
        should be added to the filter
        """
        if filterElement.objective >= self.objective and filterElement.feasible >= self.feasible:
            return -1
        if filterElement.objective <= self.objective and filterElement.feasible <= self.feasible:
            return 1
        return 0