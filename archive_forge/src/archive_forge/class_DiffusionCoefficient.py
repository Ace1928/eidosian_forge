import numpy as np
class DiffusionCoefficient:

    def __init__(self, traj, timestep, atom_indices=None, molecule=False):
        """

        This class calculates the Diffusion Coefficient for the given Trajectory using the Einstein Equation:
        
        ..math:: \\left \\langle  \\left | r(t) - r(0) \\right | ^{2} \\right \\rangle  = 2nDt
        
        where r(t) is the position of atom at time t, n is the degrees of freedom and D is the Diffusion Coefficient
        
        Solved herein by fitting with :math:`y = mx + c`, i.e. :math:`\\frac{1}{2n} \\left \\langle  \\left | r(t) - r(0) \\right | ^{2} \\right \\rangle  = Dt`, with m = D and c = 0

        wiki : https://en.wikibooks.org/wiki/Molecular_Simulation/Diffusion_Coefficients

        Parameters:
            traj (Trajectory): 
                Trajectory of atoms objects (images)
            timestep (Float): 
                Timestep between *each image in the trajectory*, in ASE timestep units
                (For an MD simulation with timestep of N, and images written every M iterations, our timestep here is N * M) 
            atom_indices (List of Int): 
                The indices of atoms whose Diffusion Coefficient is to be calculated explicitly
            molecule (Boolean)
                Indicate if we are studying a molecule instead of atoms, therefore use centre of mass in calculations
 
        """
        self.traj = traj
        self.timestep = timestep
        self.atom_indices = atom_indices
        if self.atom_indices == None:
            self.atom_indices = [i for i in range(len(traj[0]))]
        self.is_molecule = molecule
        if self.is_molecule:
            self.types_of_atoms = ['molecule']
            self.no_of_atoms = [1]
        else:
            self.types_of_atoms = sorted(set(traj[0].symbols[self.atom_indices]))
            self.no_of_atoms = [traj[0].get_chemical_symbols().count(symbol) for symbol in self.types_of_atoms]
        self._slopes = []

    @property
    def no_of_types_of_atoms(self):
        """

        Dynamically returns the number of different atoms in the system
        
        """
        return len(self.types_of_atoms)

    @property
    def slopes(self):
        """

        Method to return slopes fitted to datapoints. If undefined, calculate slopes

        """
        if len(self._slopes) == 0:
            self.calculate()
        return self._slopes

    @slopes.setter
    def slopes(self, values):
        """
 
        Method to set slopes as fitted to datapoints

        """
        self._slopes = values

    def _initialise_arrays(self, ignore_n_images, number_of_segments):
        """

        Private function to initialise data storage objects. This includes objects to store the total timesteps
        sampled, the average diffusivity for species in any given segment, and objects to store gradient and intercept from fitting.

        Parameters:
            ignore_n_images (Int): 
                Number of images you want to ignore from the start of the trajectory, e.g. during equilibration
            number_of_segments (Int): 
                Divides the given trajectory in to segments to allow statistical analysis
        
        """
        total_images = len(self.traj) - ignore_n_images
        self.no_of_segments = number_of_segments
        self.len_segments = total_images // self.no_of_segments
        self.timesteps = np.linspace(0, total_images * self.timestep, total_images + 1)
        self.xyz_segment_ensemble_average = np.zeros((self.no_of_segments, self.no_of_types_of_atoms, 3, self.len_segments))
        self.slopes = np.zeros((self.no_of_types_of_atoms, self.no_of_segments, 3))
        self.intercepts = np.zeros((self.no_of_types_of_atoms, self.no_of_segments, 3))
        self.cont_xyz_segment_ensemble_average = 0

    def calculate(self, ignore_n_images=0, number_of_segments=1):
        """
        
        Calculate the diffusion coefficients, using the previously supplied data. The user can break the data into segments and 
        take the average over these trajectories, therefore allowing statistical analysis and derivation of standard deviations.
        Option is also provided to ignore initial images if data is perhaps unequilibrated initially.

        Parameters:
            ignore_n_images (Int): 
                Number of images you want to ignore from the start of the trajectory, e.g. during equilibration
            number_of_segments (Int): 
                Divides the given trajectory in to segments to allow statistical analysis

        """
        self._initialise_arrays(ignore_n_images, number_of_segments)
        for segment_no in range(self.no_of_segments):
            start = segment_no * self.len_segments
            end = start + self.len_segments
            seg = self.traj[ignore_n_images + start:ignore_n_images + end]
            if self.is_molecule:
                com_orig = np.zeros(3)
                for atom_no in self.atom_indices:
                    com_orig += seg[0].positions[atom_no] / len(self.atom_indices)
            for image_no in range(0, len(seg)):
                xyz_disp = np.zeros((self.no_of_types_of_atoms, 3))
                if not self.is_molecule:
                    for atom_no in self.atom_indices:
                        sym_index = self.types_of_atoms.index(seg[image_no].symbols[atom_no])
                        xyz_disp[sym_index] += np.square(seg[image_no].positions[atom_no] - seg[0].positions[atom_no])
                else:
                    com_disp = np.zeros(3)
                    for atom_no in self.atom_indices:
                        com_disp += seg[image_no].positions[atom_no] / len(self.atom_indices)
                    xyz_disp[0] += np.square(com_disp - com_orig)
                for sym_index in range(self.no_of_types_of_atoms):
                    denominator = 2 * self.no_of_atoms[sym_index]
                    for xyz in range(3):
                        self.xyz_segment_ensemble_average[segment_no][sym_index][xyz][image_no] = xyz_disp[sym_index][xyz] / denominator
            for sym_index in range(self.no_of_types_of_atoms):
                self.slopes[sym_index][segment_no], self.intercepts[sym_index][segment_no] = self._fit_data(self.timesteps[start:end], self.xyz_segment_ensemble_average[segment_no][sym_index])

    def _fit_data(self, x, y):
        """
        Private function that returns slope and intercept for linear fit to mean square diffusion data


        Parameters:
            x (Array of floats): 
                Linear list of timesteps in the calculation
            y (Array of floats): 
                Mean square displacement as a function of time.
        
        """
        slopes = np.zeros(3)
        intercepts = np.zeros(3)
        x_edited = np.vstack([np.array(x), np.ones(len(x))]).T
        for xyz in range(3):
            slopes[xyz], intercepts[xyz] = np.linalg.lstsq(x_edited, y[xyz], rcond=-1)[0]
        return (slopes, intercepts)

    def get_diffusion_coefficients(self):
        """
        
        Returns diffusion coefficients for atoms (in alphabetical order) along with standard deviation.
        
        All data is currently passed out in units of Å^2/<ASE time units>
        To convert into Å^2/fs => multiply by ase.units.fs
        To convert from Å^2/fs to cm^2/s => multiply by (10^-8)^2 / 10^-15 = 10^-1
        
        """
        slopes = [np.mean(self.slopes[sym_index]) for sym_index in range(self.no_of_types_of_atoms)]
        std = [np.std(self.slopes[sym_index]) for sym_index in range(self.no_of_types_of_atoms)]
        return (slopes, std)

    def plot(self, ax=None, show=False):
        """
        
        Auto-plot of Diffusion Coefficient data. Provides basic framework for visualising analysis.
 
         Parameters:
            ax (Matplotlib.axes.Axes)
                Axes object on to which plot can be created
            show (Boolean)
                Whether or not to show the created plot. Default: False
        
        """
        import matplotlib.pyplot as plt
        from ase.units import fs as fs_conversion
        if ax is None:
            ax = plt.gca()
        color_list = plt.cm.Set3(np.linspace(0, 1, self.no_of_types_of_atoms))
        xyz_labels = ['X', 'Y', 'Z']
        xyz_markers = ['o', 's', '^']
        graph_timesteps = self.timesteps / fs_conversion
        for segment_no in range(self.no_of_segments):
            start = segment_no * self.len_segments
            end = start + self.len_segments
            label = None
            for sym_index in range(self.no_of_types_of_atoms):
                for xyz in range(3):
                    if segment_no == 0:
                        label = 'Species: %s (%s)' % (self.types_of_atoms[sym_index], xyz_labels[xyz])
                    ax.scatter(graph_timesteps[start:end], self.xyz_segment_ensemble_average[segment_no][sym_index][xyz], color=color_list[sym_index], marker=xyz_markers[xyz], label=label, linewidth=1, edgecolor='grey')
                line = np.mean(self.slopes[sym_index][segment_no]) * fs_conversion * graph_timesteps[start:end] + np.mean(self.intercepts[sym_index][segment_no])
                if segment_no == 0:
                    label = 'Segment Mean : %s' % self.types_of_atoms[sym_index]
                ax.plot(graph_timesteps[start:end], line, color='C%d' % sym_index, label=label, linestyle='--')
            x_coord = graph_timesteps[end - 1]
            ax.plot([x_coord, x_coord], [-0.001, 1.05 * np.amax(self.xyz_segment_ensemble_average)], color='grey', linestyle=':')
        ax.set_ylim(-0.001, 1.05 * np.amax(self.xyz_segment_ensemble_average))
        ax.legend(loc='best')
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Mean Square Displacement ($\\AA^2$)')
        if show:
            plt.show()

    def print_data(self):
        """
        
        Output of statistical analysis for Diffusion Coefficient data. Provides basic framework for understanding calculation.
        
        """
        from ase.units import fs as fs_conversion
        slopes, std = self.get_diffusion_coefficients()
        for sym_index in range(self.no_of_types_of_atoms):
            print('---')
            print('Species: %4s' % self.types_of_atoms[sym_index])
            print('---')
            for segment_no in range(self.no_of_segments):
                print('Segment   %3d:         Diffusion Coefficient = %.10f Å^2/fs; Intercept = %.10f Å^2;' % (segment_no, np.mean(self.slopes[sym_index][segment_no]) * fs_conversion, np.mean(self.intercepts[sym_index][segment_no])))
        print('---')
        for sym_index in range(self.no_of_types_of_atoms):
            print('Mean Diffusion Coefficient (X, Y and Z) : %s = %.10f Å^2/fs; Std. Dev. = %.10f Å^2/fs' % (self.types_of_atoms[sym_index], slopes[sym_index] * fs_conversion, std[sym_index] * fs_conversion))
        print('---')