from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .. import constants
import os
import warnings
class CPLEX_CMD(LpSolver_CMD):
    """The CPLEX LP solver"""
    name = 'CPLEX_CMD'

    def __init__(self, mip=True, msg=True, timeLimit=None, gapRel=None, gapAbs=None, options=None, warmStart=False, keepFiles=False, path=None, threads=None, logPath=None, maxMemory=None, maxNodes=None):
        """
        :param bool mip: if False, assume LP even if integer variables
        :param bool msg: if False, no log is shown
        :param float timeLimit: maximum time for solver (in seconds)
        :param float gapRel: relative gap tolerance for the solver to stop (in fraction)
        :param float gapAbs: absolute gap tolerance for the solver to stop
        :param int threads: sets the maximum number of threads
        :param list options: list of additional options to pass to solver
        :param bool warmStart: if True, the solver will use the current value of variables as a start
        :param bool keepFiles: if True, files are saved in the current directory and not deleted after solving
        :param str path: path to the solver binary
        :param str logPath: path to the log file
        :param float maxMemory: max memory to use during the solving. Stops the solving when reached.
        :param int maxNodes: max number of nodes during branching. Stops the solving when reached.
        """
        LpSolver_CMD.__init__(self, gapRel=gapRel, mip=mip, msg=msg, timeLimit=timeLimit, options=options, maxMemory=maxMemory, maxNodes=maxNodes, warmStart=warmStart, path=path, keepFiles=keepFiles, threads=threads, gapAbs=gapAbs, logPath=logPath)

    def defaultPath(self):
        return self.executableExtension('cplex')

    def available(self):
        """True if the solver is available"""
        return self.executable(self.path)

    def actualSolve(self, lp):
        """Solve a well formulated lp problem"""
        if not self.executable(self.path):
            raise PulpSolverError('PuLP: cannot execute ' + self.path)
        tmpLp, tmpSol, tmpMst = self.create_tmp_files(lp.name, 'lp', 'sol', 'mst')
        vs = lp.writeLP(tmpLp, writeSOS=1)
        try:
            os.remove(tmpSol)
        except:
            pass
        if not self.msg:
            cplex = subprocess.Popen(self.path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            cplex = subprocess.Popen(self.path, stdin=subprocess.PIPE)
        cplex_cmds = 'read ' + tmpLp + '\n'
        if self.optionsDict.get('warmStart', False):
            self.writesol(filename=tmpMst, vs=vs)
            cplex_cmds += 'read ' + tmpMst + '\n'
            cplex_cmds += 'set advance 1\n'
        if self.timeLimit is not None:
            cplex_cmds += 'set timelimit ' + str(self.timeLimit) + '\n'
        options = self.options + self.getOptions()
        for option in options:
            cplex_cmds += option + '\n'
        if lp.isMIP():
            if self.mip:
                cplex_cmds += 'mipopt\n'
                cplex_cmds += 'change problem fixed\n'
            else:
                cplex_cmds += 'change problem lp\n'
        cplex_cmds += 'optimize\n'
        cplex_cmds += 'write ' + tmpSol + '\n'
        cplex_cmds += 'quit\n'
        cplex_cmds = cplex_cmds.encode('UTF-8')
        cplex.communicate(cplex_cmds)
        if cplex.returncode != 0:
            raise PulpSolverError('PuLP: Error while trying to execute ' + self.path)
        if not os.path.exists(tmpSol):
            status = constants.LpStatusInfeasible
            values = reducedCosts = shadowPrices = slacks = solStatus = None
        else:
            status, values, reducedCosts, shadowPrices, slacks, solStatus = self.readsol(tmpSol)
        self.delete_tmp_files(tmpLp, tmpMst, tmpSol)
        if self.optionsDict.get('logPath') != 'cplex.log':
            self.delete_tmp_files('cplex.log')
        if status != constants.LpStatusInfeasible:
            lp.assignVarsVals(values)
            lp.assignVarsDj(reducedCosts)
            lp.assignConsPi(shadowPrices)
            lp.assignConsSlack(slacks)
        lp.assignStatus(status, solStatus)
        return status

    def getOptions(self):
        params_eq = dict(logPath='set logFile {}', gapRel='set mip tolerances mipgap {}', gapAbs='set mip tolerances absmipgap {}', maxMemory='set mip limits treememory {}', threads='set threads {}', maxNodes='set mip limits nodes {}')
        return [v.format(self.optionsDict[k]) for k, v in params_eq.items() if k in self.optionsDict and self.optionsDict[k] is not None]

    def readsol(self, filename):
        """Read a CPLEX solution file"""
        try:
            import xml.etree.ElementTree as et
        except ImportError:
            import elementtree.ElementTree as et
        solutionXML = et.parse(filename).getroot()
        solutionheader = solutionXML.find('header')
        statusString = solutionheader.get('solutionStatusString')
        statusValue = solutionheader.get('solutionStatusValue')
        cplexStatus = {'1': constants.LpStatusOptimal, '101': constants.LpStatusOptimal, '102': constants.LpStatusOptimal, '104': constants.LpStatusOptimal, '105': constants.LpStatusOptimal, '107': constants.LpStatusOptimal, '109': constants.LpStatusOptimal, '113': constants.LpStatusOptimal}
        if statusValue not in cplexStatus:
            raise PulpSolverError("Unknown status returned by CPLEX: \ncode: '{}', string: '{}'".format(statusValue, statusString))
        status = cplexStatus[statusValue]
        cplexSolStatus = {'104': constants.LpSolutionIntegerFeasible, '105': constants.LpSolutionIntegerFeasible, '107': constants.LpSolutionIntegerFeasible, '109': constants.LpSolutionIntegerFeasible, '111': constants.LpSolutionIntegerFeasible, '113': constants.LpSolutionIntegerFeasible}
        solStatus = cplexSolStatus.get(statusValue)
        shadowPrices = {}
        slacks = {}
        constraints = solutionXML.find('linearConstraints')
        for constraint in constraints:
            name = constraint.get('name')
            slack = constraint.get('slack')
            shadowPrice = constraint.get('dual')
            try:
                shadowPrices[name] = float(shadowPrice)
            except TypeError:
                shadowPrices[name] = None
            slacks[name] = float(slack)
        values = {}
        reducedCosts = {}
        for variable in solutionXML.find('variables'):
            name = variable.get('name')
            value = variable.get('value')
            values[name] = float(value)
            reducedCost = variable.get('reducedCost')
            try:
                reducedCosts[name] = float(reducedCost)
            except TypeError:
                reducedCosts[name] = None
        return (status, values, reducedCosts, shadowPrices, slacks, solStatus)

    def writesol(self, filename, vs):
        """Writes a CPLEX solution file"""
        try:
            import xml.etree.ElementTree as et
        except ImportError:
            import elementtree.ElementTree as et
        root = et.Element('CPLEXSolution', version='1.2')
        attrib_head = dict()
        attrib_quality = dict()
        et.SubElement(root, 'header', attrib=attrib_head)
        et.SubElement(root, 'header', attrib=attrib_quality)
        variables = et.SubElement(root, 'variables')
        values = [(v.name, v.value()) for v in vs if v.value() is not None]
        for index, (name, value) in enumerate(values):
            attrib_vars = dict(name=name, value=str(value), index=str(index))
            et.SubElement(variables, 'variable', attrib=attrib_vars)
        mst = et.ElementTree(root)
        mst.write(filename, encoding='utf-8', xml_declaration=True)
        return True