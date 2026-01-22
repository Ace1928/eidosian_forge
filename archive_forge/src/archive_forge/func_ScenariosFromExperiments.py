import pyomo.environ as pyo
def ScenariosFromExperiments(self, addtoSet):
    """Creates new self.Scenarios list using the experiments only.

        Args:
            addtoSet (ScenarioSet): the scenarios will be added to this set
        Returns:
            a ScenarioSet
        """
    assert isinstance(addtoSet, ScenarioSet)
    scenario_numbers = list(range(len(self.pest.callback_data)))
    prob = 1.0 / len(scenario_numbers)
    for exp_num in scenario_numbers:
        model = self.pest._instance_creation_callback(exp_num, self.pest.callback_data)
        opt = pyo.SolverFactory(self.solvername)
        results = opt.solve(model)
        ThetaVals = dict()
        for theta in self.pest.theta_names:
            tvar = eval('model.' + theta)
            tval = pyo.value(tvar)
            ThetaVals[theta] = tval
        addtoSet.addone(ParmestScen('ExpScen' + str(exp_num), ThetaVals, prob))