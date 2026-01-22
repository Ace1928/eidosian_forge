import pyomo.environ as pyo
class ScenarioSet(object):
    """
    Class to hold scenario sets

    Args:
    name (str): name of the set (might be "")

    """

    def __init__(self, name):
        self._scens = list()
        self.name = name

    def _firstscen(self):
        assert len(self._scens) > 0
        return self._scens[0]

    def ScensIterator(self):
        """Usage: for scenario in ScensIterator()"""
        return iter(self._scens)

    def ScenarioNumber(self, scennum):
        """Returns the scenario with the given, zero-based number"""
        return self._scens[scennum]

    def addone(self, scen):
        """Add a scenario to the set

        Args:
            scen (ParmestScen): the scenario to add
        """
        assert isinstance(self._scens, list)
        self._scens.append(scen)

    def append_bootstrap(self, bootstrap_theta):
        """Append a bootstrap theta df to the scenario set; equally likely

        Args:
            bootstrap_theta (dataframe): created by the bootstrap
        Note: this can be cleaned up a lot with the list becomes a df,
              which is why I put it in the ScenarioSet class.
        """
        assert len(bootstrap_theta) > 0
        prob = 1.0 / len(bootstrap_theta)
        dfdict = bootstrap_theta.to_dict(orient='index')
        for index, ThetaVals in dfdict.items():
            name = 'Bootstrap' + str(index)
            self.addone(ParmestScen(name, ThetaVals, prob))

    def write_csv(self, filename):
        """write a csv file with the scenarios in the set

        Args:
            filename (str): full path and full name of file
        """
        if len(self._scens) == 0:
            print('Empty scenario set, not writing file={}'.format(filename))
            return
        with open(filename, 'w') as f:
            f.write('Name,Probability')
            for n in self._firstscen().ThetaVals.keys():
                f.write(',{}'.format(n))
            f.write('\n')
            for s in self.ScensIterator():
                f.write('{},{}'.format(s.name, s.probability))
                for v in s.ThetaVals.values():
                    f.write(',{}'.format(v))
                f.write('\n')