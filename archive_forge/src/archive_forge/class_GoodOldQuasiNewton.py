import time
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
class GoodOldQuasiNewton(Optimizer):

    def __init__(self, atoms, restart=None, logfile='-', trajectory=None, fmax=None, converged=None, hessianupdate='BFGS', hessian=None, forcemin=True, verbosity=None, maxradius=None, diagonal=20.0, radius=None, transitionstate=False, master=None):
        """Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            File used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            File used to store trajectory of atomic movement.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2 Angstroms).


        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.
        """
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)
        self.eps = 1e-12
        self.hessianupdate = hessianupdate
        self.forcemin = forcemin
        self.verbosity = verbosity
        self.diagonal = diagonal
        self.atoms = atoms
        n = len(self.atoms) * 3
        if radius is None:
            self.radius = 0.05 * np.sqrt(n) / 10.0
        else:
            self.radius = radius
        if maxradius is None:
            self.maxradius = 0.5 * np.sqrt(n)
        else:
            self.maxradius = maxradius
        self.radius = max(min(self.radius, self.maxradius), 0.0001)
        self.transitionstate = transitionstate
        if hasattr(atoms, 'springconstant'):
            self.forcemin = False
        self.t0 = time.time()

    def initialize(self):
        pass

    def write_log(self, text):
        if self.logfile is not None:
            self.logfile.write(text + '\n')
            self.logfile.flush()

    def set_hessian(self, hessian):
        self.hessian = hessian

    def get_hessian(self):
        if not hasattr(self, 'hessian'):
            self.set_default_hessian()
        return self.hessian

    def set_default_hessian(self):
        n = len(self.atoms) * 3
        hessian = np.zeros((n, n))
        for i in range(n):
            hessian[i][i] = self.diagonal
        self.set_hessian(hessian)

    def update_hessian(self, pos, G):
        import copy
        if hasattr(self, 'oldG'):
            if self.hessianupdate == 'BFGS':
                self.update_hessian_bfgs(pos, G)
            elif self.hessianupdate == 'Powell':
                self.update_hessian_powell(pos, G)
            else:
                self.update_hessian_bofill(pos, G)
        elif not hasattr(self, 'hessian'):
            self.set_default_hessian()
        self.oldpos = copy.copy(pos)
        self.oldG = copy.copy(G)
        if self.verbosity:
            print('hessian ', self.hessian)

    def update_hessian_bfgs(self, pos, G):
        n = len(self.hessian)
        dgrad = G - self.oldG
        dpos = pos - self.oldpos
        dotg = np.dot(dgrad, dpos)
        tvec = np.dot(dpos, self.hessian)
        dott = np.dot(dpos, tvec)
        if abs(dott) > self.eps and abs(dotg) > self.eps:
            for i in range(n):
                for j in range(n):
                    h = dgrad[i] * dgrad[j] / dotg - tvec[i] * tvec[j] / dott
                    self.hessian[i][j] += h

    def update_hessian_powell(self, pos, G):
        n = len(self.hessian)
        dgrad = G - self.oldG
        dpos = pos - self.oldpos
        absdpos = np.dot(dpos, dpos)
        if absdpos < self.eps:
            return
        dotg = np.dot(dgrad, dpos)
        tvec = dgrad - np.dot(dpos, self.hessian)
        tvecdpos = np.dot(tvec, dpos)
        ddot = tvecdpos / absdpos
        dott = np.dot(dpos, tvec)
        if abs(dott) > self.eps and abs(dotg) > self.eps:
            for i in range(n):
                for j in range(n):
                    h = tvec[i] * dpos[j] + dpos[i] * tvec[j] - ddot * dpos[i] * dpos[j]
                    h *= 1.0 / absdpos
                    self.hessian[i][j] += h

    def update_hessian_bofill(self, pos, G):
        print('update Bofill')
        n = len(self.hessian)
        dgrad = G - self.oldG
        dpos = pos - self.oldpos
        absdpos = np.dot(dpos, dpos)
        if absdpos < self.eps:
            return
        dotg = np.dot(dgrad, dpos)
        tvec = dgrad - np.dot(dpos, self.hessian)
        tvecdot = np.dot(tvec, tvec)
        tvecdpos = np.dot(tvec, dpos)
        coef1 = 1.0 - tvecdpos * tvecdpos / (absdpos * tvecdot)
        coef2 = (1.0 - coef1) * absdpos / tvecdpos
        coef3 = coef1 * tvecdpos / absdpos
        dott = np.dot(dpos, tvec)
        if abs(dott) > self.eps and abs(dotg) > self.eps:
            for i in range(n):
                for j in range(n):
                    h = coef1 * (tvec[i] * dpos[j] + dpos[i] * tvec[j]) - dpos[i] * dpos[j] * coef3 + coef2 * tvec[i] * tvec[j]
                    h *= 1.0 / absdpos
                    self.hessian[i][j] += h

    def step(self, f=None):
        """ Do one QN step
        """
        if f is None:
            f = self.atoms.get_forces()
        pos = self.atoms.get_positions().ravel()
        G = -self.atoms.get_forces().ravel()
        energy = self.atoms.get_potential_energy()
        if hasattr(self, 'oldenergy'):
            self.write_log('energies ' + str(energy) + ' ' + str(self.oldenergy))
            if self.forcemin:
                de = 0.0001
            else:
                de = 0.01
            if self.transitionstate:
                de = 0.2
            if energy - self.oldenergy > de:
                self.write_log('reject step')
                self.atoms.set_positions(self.oldpos.reshape((-1, 3)))
                G = self.oldG
                energy = self.oldenergy
                self.radius *= 0.5
            else:
                self.update_hessian(pos, G)
                de = energy - self.oldenergy
                f = 1.0
                if self.forcemin:
                    self.write_log('energy change; actual: %f estimated: %f ' % (de, self.energy_estimate))
                    if abs(self.energy_estimate) > self.eps:
                        f = abs(de / self.energy_estimate - 1)
                        self.write_log('Energy prediction factor ' + str(f))
                        self.radius *= scale_radius_energy(f, self.radius)
                else:
                    self.write_log('energy change; actual: %f ' % de)
                    self.radius *= 1.5
                fg = self.get_force_prediction(G)
                self.write_log('Scale factors %f %f ' % (scale_radius_energy(f, self.radius), scale_radius_force(fg, self.radius)))
            self.radius = max(min(self.radius, self.maxradius), 0.0001)
        else:
            self.update_hessian(pos, G)
        self.write_log('new radius %f ' % self.radius)
        self.oldenergy = energy
        b, V = eigh(self.hessian)
        V = V.T.copy()
        self.V = V
        Gbar = np.dot(G, np.transpose(V))
        lamdas = self.get_lambdas(b, Gbar)
        D = -Gbar / (b - lamdas)
        n = len(D)
        step = np.zeros(n)
        for i in range(n):
            step += D[i] * V[i]
        pos = self.atoms.get_positions().ravel()
        pos += step
        energy_estimate = self.get_energy_estimate(D, Gbar, b)
        self.energy_estimate = energy_estimate
        self.gbar_estimate = self.get_gbar_estimate(D, Gbar, b)
        self.old_gbar = Gbar
        self.atoms.set_positions(pos.reshape((-1, 3)))

    def get_energy_estimate(self, D, Gbar, b):
        de = 0.0
        for n in range(len(D)):
            de += D[n] * Gbar[n] + 0.5 * D[n] * b[n] * D[n]
        return de

    def get_gbar_estimate(self, D, Gbar, b):
        gbar_est = D * b + Gbar
        self.write_log('Abs Gbar estimate ' + str(np.dot(gbar_est, gbar_est)))
        return gbar_est

    def get_lambdas(self, b, Gbar):
        lamdas = np.zeros(len(b))
        D = -Gbar / b
        absD = np.sqrt(np.dot(D, D))
        eps = 1e-12
        nminus = self.get_hessian_inertia(b)
        if absD < self.radius:
            if not self.transitionstate:
                self.write_log('Newton step')
                return lamdas
            elif nminus == 1:
                self.write_log('Newton step')
                return lamdas
            else:
                self.write_log('Wrong inertia of Hessian matrix: %2.2f %2.2f ' % (b[0], b[1]))
        else:
            self.write_log('Corrected Newton step: abs(D) = %2.2f ' % absD)
        if not self.transitionstate:
            upperlimit = min(0, b[0]) - eps
            lamda = find_lamda(upperlimit, Gbar, b, self.radius)
            lamdas += lamda
        else:
            upperlimit = min(-b[0], b[1], 0) - eps
            lamda = find_lamda(upperlimit, Gbar, b, self.radius)
            lamdas += lamda
            lamdas[0] -= 2 * lamda
        return lamdas

    def get_hessian_inertia(self, eigenvalues):
        self.write_log('eigenvalues %2.2f %2.2f %2.2f ' % (eigenvalues[0], eigenvalues[1], eigenvalues[2]))
        n = 0
        while eigenvalues[n] < 0:
            n += 1
        return n

    def get_force_prediction(self, G):
        Gbar = np.dot(G, np.transpose(self.V))
        dGbar_actual = Gbar - self.old_gbar
        dGbar_predicted = Gbar - self.gbar_estimate
        f = np.dot(dGbar_actual, dGbar_predicted) / np.dot(dGbar_actual, dGbar_actual)
        self.write_log('Force prediction factor ' + str(f))
        return f