import pyomo.environ as pyo
def ScenariosFromBootstrap(self, addtoSet, numtomake, seed=None):
    """Creates new self.Scenarios list using the experiments only.

        Args:
            addtoSet (ScenarioSet): the scenarios will be added to this set
            numtomake (int) : number of scenarios to create
        """
    assert isinstance(addtoSet, ScenarioSet)
    bootstrap_thetas = self.pest.theta_est_bootstrap(numtomake, seed=seed)
    addtoSet.append_bootstrap(bootstrap_thetas)