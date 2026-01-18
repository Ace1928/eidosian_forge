"""
GeneParticles: Advanced Cellular Automata with Dynamic Gene Expression, Emergent Behaviors, 
and Extended Complexity
-------------------------------------------------------------------------------------------------
A hyper-advanced particle simulation that models cellular-like entities ("particles") endowed with 
complex dynamic genetic traits, adaptive behaviors, emergent properties, hierarchical speciation, 
and intricate interaction networks spanning multiple dimensions of trait synergy and competition.

This code has been enhanced to an extreme degree while preserving all original logic and complexity.
No simplifications, deletions, or truncations have been made. Instead, every aspect has been refined, 
expanded, and elaborated upon to push the system's complexity, adaptability, and performance beyond 
previous bounds. The result is a deeply parameterized, highly modular codebase capable of exhibiting 
sophisticated emergent behaviors, complex evolutionary patterns, and advanced genetic, ecological, 
and morphological dynamics.

Core Features (Now Significantly Expanded):
------------------------------------------
1. Dynamic Gene Expression & Hyper-Complex Heredity
   - Multiple mutable traits: speed, interaction strength, perception range, reproduction rate, 
     energy efficiency, synergy affinity, colony-forming propensity, evolutionary drift factors.
   - Hierarchical gene clusters with layered mutation strategies (base mutation + adaptive mutation 
     influenced by environmental conditions).
   - Nonlinear genotype-to-phenotype mappings that incorporate multiplicative and additive factors,
     epistatic interactions, and environmental feedback loops.

2. Adaptive Population Management & Advanced Homeostasis
   - Real-time FPS monitoring with multi-tiered optimization triggers.
   - Dynamic culling not only by age but also by multi-factor fitness functions involving complexity, 
     speciation stability, lineage rarity, and energy flow metrics.
   - Population growth stimulation when resources abound and multicellular colony formation 
     triggers adaptive expansions.

3. Enhanced Evolutionary Mechanisms & Deep Speciation
   - Natural selection with resource competition and survival constraints influenced by synergy networks.
   - Speciation events triggered by multidimensional genetic drift and advanced phylogenetic distance metrics.
   - Lineage tracking, with phylogenetic trees updated at intervals, integrating gene flow and mutation patterns.

4. Complex Interactions at Multiple Scales
   - Force-based dynamics with potential, gravitational, and synergy-based forces.
   - Intricate energy and mass transfer mechanics, now extended with conditional energy routing 
     based on species alliances and colony membership.
   - Emergent flocking, predation, symbiotic, and colony-like behaviors, now augmented by hierarchical 
     clustering algorithms, including multi-level KD-trees for adaptive neighborhood scaling.
   - Extended synergy matrices that change over time, influenced by environmental cues and global parameters.

5. Extreme Performance Optimization
   - Advanced vectorized operations using NumPy for all computations.
   - Multi-level spatial partitioning (KD-trees and optional R-trees or spatial hashing if desired).
   - Adaptive rendering and state management, parameterized update frequencies, and caching mechanisms 
     for recurrent computations.
   - Intricate load balancing and optional parallelization hooks (not implemented by default but structured for it).

6. Extended Configuration & Parameterization
   - Centralized configuration with extensive parameters controlling every aspect of simulation complexity.
   - Nested configuration classes for genetic parameters, interaction coefficients, evolutionary intervals, 
     synergy evolution rates, and colony formation probabilities.
   - Enhanced flexibility: all previously hard-coded values now parameterized or adjustable through config.

7. Comprehensive Documentation & Inline Comments
   - Extensive docstrings for all classes and methods.
   - Inline comments explaining complex steps, logic, and decision-making processes.
   - Maintained and expanded documentation reflecting the new complexity.

Technical Requirements:
---------------------
- Python 3.8+
- NumPy >= 1.20.0
- Pygame >= 2.0.0
- SciPy >= 1.7.0

Installation:
------------
pip install numpy pygame scipy

Usage:
------
python geneparticles.py

Controls:
- ESC: Exit simulation
"""

import math
import random
import collections
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional, Union
import time
import numpy as np
import pygame
from scipy.spatial import cKDTree

###############################################################
# Extended Configuration Classes with Nested Parameters
###############################################################

class GeneticParamConfig:
    """
    Holds genetic parameters and mutation ranges for a wide array of traits.
    This is a dedicated config structure to handle complex genetic aspects.

    Additional Traits Introduced:
    - synergy_affinity: how strongly a particle engages in synergy
    - colony_factor: how likely a particle is to form or join colonies
    - drift_sensitivity: how sensitive the particle is to evolutionary drift
    """

    def __init__(self):
        # Core gene traits and ranges
        self.gene_traits: List[str] = [
            "speed_factor", "interaction_strength", "perception_range", "reproduction_rate",
            "synergy_affinity", "colony_factor", "drift_sensitivity"
        ]

        # Increased mutation rates and ranges for more dynamic evolution
        self.gene_mutation_rate: float = 0.25  # Increased from 0.15
        self.gene_mutation_range: Tuple[float, float] = (-0.2, 0.2)  # Wider range

        # Expanded trait ranges for more diverse behaviors
        self.speed_factor_range: Tuple[float, float] = (0.05, 4.0)
        self.interaction_strength_range: Tuple[float, float] = (0.05, 4.0)
        self.perception_range_range: Tuple[float, float] = (20.0, 400.0)
        self.reproduction_rate_range: Tuple[float, float] = (0.02, 1.5)
        self.synergy_affinity_range: Tuple[float, float] = (0.0, 3.0)
        self.colony_factor_range: Tuple[float, float] = (0.0, 2.0)
        self.drift_sensitivity_range: Tuple[float, float] = (0.0, 3.0)

        # Higher energy efficiency mutation for more dynamic resource management
        self.energy_efficiency_mutation_rate: float = 0.2
        self.energy_efficiency_mutation_range: Tuple[float, float] = (-0.15, 0.3)

    def clamp_gene_values(
        self, 
        speed_factor: np.ndarray, 
        interaction_strength: np.ndarray, 
        perception_range: np.ndarray, 
        reproduction_rate: np.ndarray, 
        synergy_affinity: np.ndarray, 
        colony_factor: np.ndarray, 
        drift_sensitivity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Clamp all gene values to their specified ranges to prevent extreme values that break the simulation.
        """
        speed_factor = np.clip(speed_factor, self.speed_factor_range[0], self.speed_factor_range[1])
        interaction_strength = np.clip(interaction_strength, self.interaction_strength_range[0], self.interaction_strength_range[1])
        perception_range = np.clip(perception_range, self.perception_range_range[0], self.perception_range_range[1])
        reproduction_rate = np.clip(reproduction_rate, self.reproduction_rate_range[0], self.reproduction_rate_range[1])
        synergy_affinity = np.clip(synergy_affinity, self.synergy_affinity_range[0], self.synergy_affinity_range[1])
        colony_factor = np.clip(colony_factor, self.colony_factor_range[0], self.colony_factor_range[1])
        drift_sensitivity = np.clip(drift_sensitivity, self.drift_sensitivity_range[0], self.drift_sensitivity_range[1])
        return speed_factor, interaction_strength, perception_range, reproduction_rate, synergy_affinity, colony_factor, drift_sensitivity


class SimulationConfig:
    """
    Configuration class for the GeneParticles simulation, optimized for maximum emergence
    and complex structure formation.
    """

    def __init__(self):
        # Core simulation parameters optimized for emergence
        self.n_cell_types: int = 20  # Increased species count
        self.particles_per_type: int = 50  # Start smaller per type
        self.min_particles_per_type: int = 50  # Lower minimum for more dynamics
        self.max_particles_per_type: int = 300  # Capped for total max of 3000
        self.mass_range: Tuple[float, float] = (0.2, 15.0)  # Wider mass range
        self.base_velocity_scale: float = 0.8  # Slightly higher base speed
        self.mass_based_fraction: float = 0.7  # More mass-based interactions
        self.interaction_strength_range: Tuple[float, float] = (-3.0, 3.0)  # Wider interaction range
        self.max_frames: int = 0  # Infinite simulation
        self.initial_energy: float = 150.0  # More initial energy
        self.friction: float = 0.25  # Higher viscosity for better structure formation
        self.global_temperature: float = 0.05  # Slightly higher noise
        self.predation_range: float = 75.0  # Increased interaction range
        self.energy_transfer_factor: float = 0.7  # More efficient energy transfer
        self.mass_transfer: bool = True
        self.max_age: float = np.inf
        self.evolution_interval: int = 20000  # More frequent evolution
        self.synergy_range: float = 200.0  # Larger synergy range

        # Balanced culling weights
        self.culling_fitness_weights: Dict[str, float] = {
            "energy_weight": 0.6,
            "age_weight": 0.8,
            "speed_factor_weight": 0.7,
            "interaction_strength_weight": 0.7,
            "synergy_affinity_weight": 0.8,
            "colony_factor_weight": 0.9,
            "drift_sensitivity_weight": 0.6
        }

        # Reproduction parameters for dynamic population
        self.reproduction_energy_threshold: float = 180.0
        self.reproduction_mutation_rate: float = 0.3
        self.reproduction_offspring_energy_fraction: float = 0.5

        # Enhanced clustering parameters
        self.alignment_strength: float = 0.1  # Stronger alignment
        self.cohesion_strength: float = 0.8  # Strong cohesion
        self.separation_strength: float = 0.3  # Weaker separation
        self.cluster_radius: float = 10.0  # Tighter clusters

        self.particle_size: float = 3.0

        self.energy_efficiency_range: Tuple[float, float] = (-0.4, 3.0)

        self.genetics = GeneticParamConfig()

        # Enhanced speciation and colony parameters
        self.speciation_threshold: float = 0.8  # Easier speciation
        self.colony_formation_probability: float = 0.4  # Higher colony formation chance
        self.colony_radius: float = 250.0  # Larger colonies
        self.colony_cohesion_strength: float = 0.8  # Strong colony cohesion

        # Advanced parameters for emergence
        self.synergy_evolution_rate: float = 0.08  # Faster synergy evolution
        self.complexity_factor: float = 2.0  # Higher complexity
        self.structural_complexity_weight: float = 0.9  # Stronger emphasis on structure

        self._validate()

    def _validate(self) -> None:
        """
        Validate configuration parameters. This ensures no illegal values 
        slip through and break the simulation.
        """
        if self.n_cell_types <= 0:
            raise ValueError("Number of cell types must be greater than 0.")
        
        if self.particles_per_type <= 0:
            raise ValueError("Particles per type must be greater than 0.")
        
        if self.mass_range[0] <= 0:
            raise ValueError("Minimum mass must be positive.")
        
        if self.base_velocity_scale <= 0:
            raise ValueError("Base velocity scale must be positive.")
        
        if not (0.0 <= self.mass_based_fraction <= 1.0):
            raise ValueError("Mass-based fraction must be between 0.0 and 1.0.")
        
        if self.interaction_strength_range[0] >= self.interaction_strength_range[1]:
            raise ValueError("Interaction strength range must have the first value less than the second.")
        
        if self.max_frames < 0:
            raise ValueError("Maximum frames must be non-negative.")
        
        if self.initial_energy <= 0:
            raise ValueError("Initial energy must be positive.")
        
        if not (0.0 <= self.friction <= 1.0):
            raise ValueError("Friction must be between 0.0 and 1.0.")
        
        if self.global_temperature < 0:
            raise ValueError("Global temperature must be non-negative.")
        
        if self.predation_range <= 0:
            raise ValueError("Predation range must be positive.")
        
        if not (0.0 <= self.energy_transfer_factor <= 1.0):
            raise ValueError("Energy transfer factor must be between 0.0 and 1.0.")
        
        if self.cluster_radius <= 0:
            raise ValueError("Cluster radius must be positive.")
        
        if self.particle_size <= 0:
            raise ValueError("Particle size must be positive.")
        
        if self.speciation_threshold <= 0:
            raise ValueError("Speciation threshold must be positive.")
        
        if self.synergy_range <= 0:
            raise ValueError("Synergy range must be positive.")
        
        if self.colony_radius <= 0:
            raise ValueError("Colony radius must be positive.")
        
        if self.reproduction_energy_threshold <= 0:
            raise ValueError("Reproduction energy threshold must be positive.")
        
        if not (0.0 <= self.reproduction_offspring_energy_fraction <= 1.0):
            raise ValueError("Reproduction offspring energy fraction must be between 0.0 and 1.0.")
        
        if not (0.0 <= self.genetics.gene_mutation_rate <= 1.0):
            raise ValueError("Gene mutation rate must be between 0.0 and 1.0.")
        
        if self.genetics.gene_mutation_range[0] >= self.genetics.gene_mutation_range[1]:
            raise ValueError("Gene mutation range must have the first value less than the second.")
        
        if self.energy_efficiency_range[0] >= self.energy_efficiency_range[1]:
            raise ValueError("Energy efficiency range must have the first value less than the second.")
        
        if self.genetics.energy_efficiency_mutation_range[0] >= self.genetics.energy_efficiency_mutation_range[1]:
            raise ValueError("Energy efficiency mutation range must have the first value less than the second.")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration parameters to a dictionary for debugging or logging.
        """
        config_dict = self.__dict__.copy()
        # The genetics is a nested config; handle separately
        genetics_dict = {k: v for k, v in self.genetics.__dict__.items()}
        config_dict["genetics"] = genetics_dict
        return config_dict

###############################################################
# Utility Functions
###############################################################

def random_xy(window_width: int, window_height: int, n: int = 1) -> np.ndarray:
    """
    Generate n random (x, y) coordinates within the range [0, window_width] x [0, window_height].
    Uses NumPy for efficient generation of floating-point coordinates.

    Parameters:
    -----------
    window_width : int
        The width of the simulation window in pixels. Must be > 0.
    window_height : int
        The height of the simulation window in pixels. Must be > 0.
    n : int, optional
        The number of random coordinates to generate (default is 1). Must be > 0.

    Returns:
    --------
    np.ndarray
        A NumPy array of shape (n, 2) containing random (x, y) positions as floats.
    """
    # Validate input parameters
    assert window_width > 0, "window_width must be a positive integer."
    assert window_height > 0, "window_height must be a positive integer."
    assert n > 0, "n must be a positive integer."
    
    # Generate random coordinates using NumPy's uniform distribution
    return np.random.uniform(0, [window_width, window_height], (n, 2)).astype(float)

def generate_vibrant_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate n distinct vibrant neon RGB color tuples.
    Ensures maximum separation in the color space by evenly distributing hues.

    Parameters:
    -----------
    n : int
        The number of distinct colors to generate. Must be > 0.

    Returns:
    --------
    List[Tuple[int, int, int]]
        A list of n distinct RGB color tuples.
    """
    # Validate input parameter
    assert n > 0, "The number of colors to generate (n) must be a positive integer."

    colors = []  # Initialize list to hold color tuples
    for i in range(n):
        # Evenly distribute hues around the color wheel
        hue = (i / n) % 1.0  # Hue value between 0.0 and 1.0
        saturation = 1.0  # Full saturation for vibrant colors
        value = 1.0  # Full brightness for vibrant colors

        # Convert HSV to RGB
        h_i = int(hue * 6)  # Determine the sector of the color wheel
        f = hue * 6 - h_i  # Fractional part
        p = int(0 * 255)  # Since saturation=1, p=0
        q = int((1 - f) * 255)  # Intermediate value
        t = int(f * 255)  # Intermediate value
        v = int(value * 255)  # Maximum value

        # Assign RGB based on the hue sector
        if h_i == 0:
            r, g, b = v, t, p
        elif h_i == 1:
            r, g, b = q, v, p
        elif h_i == 2:
            r, g, b = p, v, t
        elif h_i == 3:
            r, g, b = p, q, v
        elif h_i == 4:
            r, g, b = t, p, v
        elif h_i == 5:
            r, g, b = v, p, q
        else:
            r, g, b = 255, 255, 255  # Fallback to white if hue is out of range

        colors.append((r, g, b))  # Append the color tuple to the list

    return colors  # Return the list of RGB color tuples


###############################################################
# Cellular Component & Type Data Management
###############################################################

class CellularTypeData:
    """
    Represents a cellular type with multiple cellular components.
    Manages positions, velocities, energy, mass, and genetic traits of components.
    """
    
    def __init__(self,
                 type_id: int,
                 color: Tuple[int, int, int],
                 n_particles: int,
                 window_width: int,
                 window_height: int,
                 initial_energy: float,
                 max_age: float = np.inf,
                 mass: Optional[float] = None,
                 base_velocity_scale: float = 1.0,
                 energy_efficiency: Optional[float] = None,
                 gene_traits: List[str] = ["speed_factor", "interaction_strength", "perception_range", "reproduction_rate", "synergy_affinity", "colony_factor", "drift_sensitivity"],
                 gene_mutation_rate: float = 0.05,
                 gene_mutation_range: Tuple[float, float] = (-0.1, 0.1),
                 min_energy: float = 0.0,
                 max_energy: float = 1000.0,
                 min_mass: float = 0.1,
                 max_mass: float = 10.0,
                 min_velocity: float = -10.0,
                 max_velocity: float = 10.0,
                 min_perception: float = 10.0,
                 max_perception: float = 300.0,
                 min_reproduction: float = 0.05,
                 max_reproduction: float = 1.0,
                 min_synergy: float = 0.0,
                 max_synergy: float = 2.0,
                 min_colony: float = 0.0,
                 max_colony: float = 1.0,
                 min_drift: float = 0.0,
                 max_drift: float = 2.0,
                 min_energy_efficiency: float = -0.3,
                 max_energy_efficiency: float = 2.5):
        """
        Initialize a CellularTypeData instance with given parameters.

        Parameters:
        -----------
        type_id : int
            Unique identifier for the cellular type.
        color : Tuple[int, int, int]
            RGB color tuple for rendering cellular components of this type.
        n_particles : int
            Initial number of cellular components in this type.
        window_width : int
            Width of the simulation window in pixels.
        window_height : int
            Height of the simulation window in pixels.
        initial_energy : float
            Initial energy assigned to each cellular component.
        max_age : float, default=np.inf
            Maximum age a cellular component can reach before dying.
        mass : Optional[float], default=None
            Mass of cellular components if the type is mass-based.
        base_velocity_scale : float, default=1.0
            Base scaling factor for initial velocities of cellular components.
        energy_efficiency : Optional[float], default=None
            Initial energy efficiency trait of cellular components. If None, randomly initialized within range.
        gene_traits : List[str], default=["speed_factor", "interaction_strength", "perception_range", "reproduction_rate", "synergy_affinity", "colony_factor", "drift_sensitivity"]
            List of gene trait names.
        gene_mutation_rate : float, default=0.05
            Base mutation rate for gene traits (0.0 to 1.0).
        gene_mutation_range : Tuple[float, float], default=(-0.1, 0.1)
            Range for gene trait mutations (units: arbitrary).
        min_energy : float, default=0.0
            Minimum allowed energy value.
        max_energy : float, default=1000.0
            Maximum allowed energy value.
        min_mass : float, default=0.1
            Minimum allowed mass value.
        max_mass : float, default=10.0
            Maximum allowed mass value.
        min_velocity : float, default=-10.0
            Minimum allowed velocity value.
        max_velocity : float, default=10.0
            Maximum allowed velocity value.
        min_perception : float, default=10.0
            Minimum allowed perception range.
        max_perception : float, default=300.0
            Maximum allowed perception range.
        min_reproduction : float, default=0.05
            Minimum allowed reproduction rate.
        max_reproduction : float, default=1.0
            Maximum allowed reproduction rate.
        min_synergy : float, default=0.0
            Minimum allowed synergy affinity.
        max_synergy : float, default=2.0
            Maximum allowed synergy affinity.
        min_colony : float, default=0.0
            Minimum allowed colony factor.
        max_colony : float, default=1.0
            Maximum allowed colony factor.
        min_drift : float, default=0.0
            Minimum allowed drift sensitivity.
        max_drift : float, default=2.0
            Maximum allowed drift sensitivity.
        """
        # Store metadata
        self.type_id: int = type_id  # Unique ID for the cellular type
        self.color: Tuple[int, int, int] = color  # RGB color for rendering
        self.mass_based: bool = (mass is not None)  # Flag indicating if type is mass-based

        # Store parameter bounds
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.min_mass = min_mass 
        self.max_mass = max_mass
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.min_perception = min_perception
        self.max_perception = max_perception
        self.min_reproduction = min_reproduction
        self.max_reproduction = max_reproduction
        self.min_synergy = min_synergy
        self.max_synergy = max_synergy
        self.min_colony = min_colony
        self.max_colony = max_colony
        self.min_drift = min_drift
        self.max_drift = max_drift
        self.min_energy_efficiency = min_energy_efficiency
        self.max_energy_efficiency = max_energy_efficiency

        # Initialize cellular component positions randomly within the window
        coords = random_xy(window_width, window_height, n_particles)  # Generate random (x,y) positions
        self.x: np.ndarray = coords[:, 0]  # X positions as float array
        self.y: np.ndarray = coords[:, 1]  # Y positions as float array

        # Initialize energy efficiency trait
        if energy_efficiency is None:
            # Randomly initialize energy efficiency within the defined range
            self.energy_efficiency: np.ndarray = np.random.uniform(
                self.min_energy_efficiency,
                self.max_energy_efficiency,
                n_particles
            )
        else:
            # Set a fixed energy efficiency if provided
            self.energy_efficiency: np.ndarray = np.full(n_particles, energy_efficiency, dtype=float)

        # Calculate velocity scaling based on energy efficiency and speed factor
        velocity_scaling = base_velocity_scale / self.energy_efficiency  # Higher efficiency -> lower speed

        # Initialize cellular component velocities with random values scaled by velocity_scaling
        self.vx: np.ndarray = np.clip(
            np.random.uniform(-0.5, 0.5, n_particles) * velocity_scaling,
            self.min_velocity, self.max_velocity
        ).astype(float)  # X velocities
        self.vy: np.ndarray = np.clip(
            np.random.uniform(-0.5, 0.5, n_particles) * velocity_scaling,
            self.min_velocity, self.max_velocity
        ).astype(float)  # Y velocities

        # Initialize energy levels for all cellular components
        self.energy: np.ndarray = np.clip(
            np.full(n_particles, initial_energy, dtype=float),
            self.min_energy, self.max_energy
        )  # Energy levels

        # Initialize mass if type is mass-based
        if self.mass_based:
            if mass is None or mass <= 0.0:
                # Mass must be positive for mass-based types
                raise ValueError("Mass must be positive for mass-based cellular types.")
            self.mass: np.ndarray = np.clip(
                np.full(n_particles, mass, dtype=float),
                self.min_mass, self.max_mass
            )  # Mass values
        else:
            self.mass = None  # Mass is None for massless types

        # Initialize alive status and age for cellular components
        self.alive: np.ndarray = np.ones(n_particles, dtype=bool)  # All components start as alive
        self.age: np.ndarray = np.zeros(n_particles, dtype=float)  # Initial age is 0
        self.max_age: float = max_age  # Maximum age before death

        # Initialize gene traits
        # Genes influence behaviors and can mutate during reproduction
        self.speed_factor: np.ndarray = np.random.uniform(0.5, 1.5, n_particles)  # Speed scaling factors
        self.interaction_strength: np.ndarray = np.random.uniform(0.5, 1.5, n_particles)  # Interaction force scaling factors
        self.perception_range: np.ndarray = np.clip(
            np.random.uniform(50.0, 150.0, n_particles),
            self.min_perception, self.max_perception
        )  # Perception ranges for interactions
        self.reproduction_rate: np.ndarray = np.clip(
            np.random.uniform(0.1, 0.5, n_particles),
            self.min_reproduction, self.max_reproduction
        )  # Reproduction rates
        self.synergy_affinity: np.ndarray = np.clip(
            np.random.uniform(0.5, 1.5, n_particles),
            self.min_synergy, self.max_synergy
        )  # Synergy affinity factors
        self.colony_factor: np.ndarray = np.clip(
            np.random.uniform(0.0, 1.0, n_particles),
            self.min_colony, self.max_colony
        )  # Colony formation factors
        self.drift_sensitivity: np.ndarray = np.clip(
            np.random.uniform(0.5, 1.5, n_particles),
            self.min_drift, self.max_drift
        )  # Evolutionary drift sensitivity

        # Gene mutation parameters
        self.gene_mutation_rate: float = gene_mutation_rate  # Mutation rate for genes
        self.gene_mutation_range: Tuple[float, float] = gene_mutation_range  # Mutation range for genes

        # Speciation parameters
        self.species_id: np.ndarray = np.full(n_particles, type_id, dtype=int)  # Species IDs, initially same as type_id

        # Lineage tracking
        self.parent_id: np.ndarray = np.full(n_particles, -1, dtype=int)  # Parent IDs (-1 indicates no parent)

        # Colony tracking
        self.colony_id: np.ndarray = np.full(n_particles, -1, dtype=int)  # Colony IDs (-1 indicates no colony)
        self.colony_role: np.ndarray = np.zeros(n_particles, dtype=int)  # Colony roles (0=none, 1=leader, 2=member)

        # Synergy network
        self.synergy_connections: np.ndarray = np.zeros((n_particles, n_particles), dtype=bool)  # Synergy connection matrix

        # Adaptation metrics
        self.fitness_score: np.ndarray = np.zeros(n_particles, dtype=float)  # Individual fitness scores
        self.generation: np.ndarray = np.zeros(n_particles, dtype=int)  # Generation counter
        self.mutation_history: List[List[Tuple[str, float]]] = [[] for _ in range(n_particles)]  # Track mutations

    def is_alive_mask(self) -> np.ndarray:
        """
        Compute a mask of alive cellular components based on various conditions:
        - Energy must be greater than 0.
        - If mass-based, mass must be greater than 0.
        - Age must be less than max_age.
        - Alive flag must be True.
        Returns:
        --------
        np.ndarray
            Boolean array indicating alive status of cellular components.
        """
        # Basic alive conditions: energy > 0, age < max_age, and alive flag is True
        mask = (self.alive & (self.energy > 0.0) & (self.age < self.max_age))
        if self.mass_based and self.mass is not None:
            # Additional condition for mass-based types: mass > 0
            mask = mask & (self.mass > 0.0)
        return mask  # Return the computed alive mask

    def update_alive(self) -> None:
        """
        Update the alive status of cellular components based on current conditions.
        """
        self.alive = self.is_alive_mask()  # Update alive mask based on current conditions

    def age_components(self) -> None:
        """
        Increment the age of each cellular component and apply a minimal energy drain due to aging.
        """
        self.age += 1.0  # Increment age by 1 unit per frame
        self.energy = np.clip(self.energy, 0.0, None)  # Ensure energy doesn't go below 0

    def update_states(self) -> None:
        """
        Update the state of cellular components if needed.
        Currently a placeholder for future expansions (e.g., 'active', 'inactive').
        """
        pass  # No state arrays are stored; this can be expanded as needed

    def remove_dead(self, config: SimulationConfig) -> None:
        """
        Remove dead cellular components from the type to keep data arrays compact.
        Filters all component attributes based on the alive mask.
        Additionally, transfers energy from components dying of old age to their nearest three living neighbors.

        Parameters:
        -----------
        config : SimulationConfig
            Configuration parameters for the simulation, needed for predation_range.
        """
        # Validate array sizes first to prevent index mismatches
        array_size = self.x.size
        if not all(getattr(self, attr).size == array_size for attr in [
            'y', 'vx', 'vy', 'energy', 'alive', 'age', 'energy_efficiency',
            'speed_factor', 'interaction_strength', 'perception_range',
            'reproduction_rate', 'synergy_affinity', 'colony_factor',
            'drift_sensitivity', 'species_id', 'parent_id'
        ]):
            # If arrays are mismatched, synchronize them to smallest size
            min_size = min(getattr(self, attr).size for attr in [
                'x', 'y', 'vx', 'vy', 'energy', 'alive', 'age', 'energy_efficiency',
                'speed_factor', 'interaction_strength', 'perception_range',
                'reproduction_rate', 'synergy_affinity', 'colony_factor',
                'drift_sensitivity', 'species_id', 'parent_id'
            ])
            for attr in [
                'x', 'y', 'vx', 'vy', 'energy', 'alive', 'age', 'energy_efficiency',
                'speed_factor', 'interaction_strength', 'perception_range',
                'reproduction_rate', 'synergy_affinity', 'colony_factor',
                'drift_sensitivity', 'species_id', 'parent_id'
            ]:
                setattr(self, attr, getattr(self, attr)[:min_size])
            if self.mass_based and self.mass is not None:
                self.mass = self.mass[:min_size]

        # Get current alive status
        alive_mask = self.is_alive_mask()
        
        # Identify components dying of old age
        dead_due_to_age = (~alive_mask) & (self.age >= self.max_age)
        
        if np.any(dead_due_to_age):
            # Find indices of alive and dead-by-age components
            alive_indices = np.where(alive_mask)[0]
            dead_age_indices = np.where(dead_due_to_age)[0]
            
            if alive_indices.size > 0:
                # Build positions array for KD-Tree
                alive_positions = np.column_stack((self.x[alive_indices], self.y[alive_indices]))
                tree = cKDTree(alive_positions)
                
                # Process energy transfer in batches for better performance
                batch_size = 1000
                for i in range(0, len(dead_age_indices), batch_size):
                    batch_indices = dead_age_indices[i:i + batch_size]
                    dead_positions = np.column_stack((self.x[batch_indices], self.y[batch_indices]))
                    dead_energies = self.energy[batch_indices]
                    
                    # Find nearest neighbors for all dead components in batch
                    distances, neighbors = tree.query(
                        dead_positions, 
                        k=3,
                        distance_upper_bound=config.predation_range
                    )
                    
                    # Process energy transfer
                    valid_mask = distances < config.predation_range
                    for j, (dist_row, neighbor_row, dead_energy) in enumerate(zip(distances, neighbors, dead_energies)):
                        valid = valid_mask[j]
                        if np.any(valid):
                            valid_neighbors = neighbor_row[valid]
                            energy_share = dead_energy / np.sum(valid)
                            self.energy[alive_indices[valid_neighbors]] += energy_share
                            self.energy[batch_indices[j]] = 0.0

        # Apply filtering to all arrays
        arrays_to_filter = [
            'x', 'y', 'vx', 'vy', 'energy', 'alive', 'age', 'energy_efficiency',
            'speed_factor', 'interaction_strength', 'perception_range',
            'reproduction_rate', 'synergy_affinity', 'colony_factor',
            'drift_sensitivity', 'species_id', 'parent_id'
        ]
        
        for attr in arrays_to_filter:
            try:
                setattr(self, attr, getattr(self, attr)[alive_mask])
            except IndexError:
                # If error occurs, resize array to match alive_mask size
                current_array = getattr(self, attr)
                setattr(self, attr, current_array[:len(alive_mask)][alive_mask])
                
        # Handle mass array separately if it exists
        if self.mass_based and self.mass is not None:
            try:
                self.mass = self.mass[alive_mask]
            except IndexError:
                self.mass = self.mass[:len(alive_mask)][alive_mask]

    def add_component(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        energy: float,
        mass_val: Optional[float],
        energy_efficiency_val: float,
        speed_factor_val: float,
        interaction_strength_val: float,
        perception_range_val: float,
        reproduction_rate_val: float,
        synergy_affinity_val: float,
        colony_factor_val: float,
        drift_sensitivity_val: float,
        species_id_val: int,
        parent_id_val: int,
        max_age: float
    ) -> None:
        """
        Add a new cellular component (e.g., offspring) to this cellular type.

        Parameters:
        -----------
        x : float
            X-coordinate of the new component.
        y : float
            Y-coordinate of the new component.
        vx : float
            X-component of the new component's velocity.
        vy : float
            Y-component of the new component's velocity.
        energy : float
            Initial energy of the new component.
        mass_val : Optional[float]
            Mass of the new component if type is mass-based; else None.
        energy_efficiency_val : float
            Energy efficiency trait of the new component.
        speed_factor_val : float
            Speed factor gene trait of the new component.
        interaction_strength_val : float
            Interaction strength gene trait of the new component.
        perception_range_val : float
            Perception range gene trait of the new component.
        reproduction_rate_val : float
            Reproduction rate gene trait of the new component.
        synergy_affinity_val : float
            Synergy affinity gene trait of the new component.
        colony_factor_val : float
            Colony formation gene trait of the new component.
        drift_sensitivity_val : float
            Drift sensitivity gene trait of the new component.
        species_id_val : int
            Species ID of the new component.
        parent_id_val : int
            Parent ID of the new component.
        max_age : float
            Maximum age for the new component.
        """
        # Append new component's attributes using NumPy's concatenate for efficiency
        self.x = np.concatenate((self.x, [x]))  # Append new x position
        self.y = np.concatenate((self.y, [y]))  # Append new y position
        self.vx = np.concatenate((self.vx, [vx]))  # Append new x velocity
        self.vy = np.concatenate((self.vy, [vy]))  # Append new y velocity
        self.energy = np.concatenate((self.energy, [energy]))  # Append new energy level
        self.alive = np.concatenate((self.alive, [True]))  # Set alive status to True
        self.age = np.concatenate((self.age, [0.0]))  # Initialize age to 0 for new component
        self.energy_efficiency = np.concatenate((self.energy_efficiency, [energy_efficiency_val]))  # Append energy efficiency
        self.speed_factor = np.concatenate((self.speed_factor, [speed_factor_val]))  # Append speed factor
        self.interaction_strength = np.concatenate((self.interaction_strength, [interaction_strength_val]))  # Append interaction strength
        self.perception_range = np.concatenate((self.perception_range, [perception_range_val]))  # Append perception range
        self.reproduction_rate = np.concatenate((self.reproduction_rate, [reproduction_rate_val]))  # Append reproduction rate
        self.synergy_affinity = np.concatenate((self.synergy_affinity, [synergy_affinity_val]))  # Append synergy affinity
        self.colony_factor = np.concatenate((self.colony_factor, [colony_factor_val]))  # Append colony factor
        self.drift_sensitivity = np.concatenate((self.drift_sensitivity, [drift_sensitivity_val]))  # Append drift sensitivity
        self.species_id = np.concatenate((self.species_id, [species_id_val]))  # Append species ID
        self.parent_id = np.concatenate((self.parent_id, [parent_id_val]))  # Append parent ID

        if self.mass_based and self.mass is not None:
            if mass_val is None or mass_val <= 0.0:
                # Ensure mass is positive; assign a small random mass if invalid
                mass_val = max(0.1, abs(mass_val if mass_val is not None else 1.0))
            self.mass = np.concatenate((self.mass, [mass_val]))  # Append new mass value

###############################################################
# Genetic Interpreter Class
###############################################################

class GeneticInterpreter:
    """
    Decodes and interprets the genetic sequence of particles.
    Genes define behavior, traits, reproduction, and interactions.
    Each gene includes redundant components and multiple mutable parts.
    """
    
    def __init__(self, gene_sequence: Optional[List[List[Any]]] = None):
        """
        Initialize the GeneticInterpreter with a gene sequence.

        Parameters:
        -----------
        gene_sequence : Optional[List[List[Any]]], default=None
            Encoded genetic data (list of symbolic gene arrays). If None, uses default sequence.
        """
        # Default gene sequence if none provided
        self.default_sequence = [
            ["start_movement", 1.0, 0.1, 0.0],  # [speed_modifier, randomness, direction_bias]
            ["start_interaction", 0.5, 100.0],   # [attraction_strength, interaction_radius]
            ["start_energy", 0.1, 0.5, 0.3],     # [passive_gain, feeding_efficiency, predation_efficiency]
            ["start_reproduction", 150.0, 100.0, 50.0, 30.0],  # [sexual_threshold, asexual_threshold, reproduction_cost, cooldown_time]
            ["start_growth", 0.1, 2.0, 100.0],   # [growth_rate, adult_size, maturity_age]
            ["start_predation", 10.0, 5.0]       # [attack_power, energy_gain]
        ]
        self.gene_sequence = gene_sequence if gene_sequence is not None else self.default_sequence

    def decode(self, particle: CellularTypeData, others: List[CellularTypeData], env: SimulationConfig) -> None:
        """
        Decodes the genetic sequence to influence particle behavior.
        Each gene's symbolic structure is mapped to corresponding traits or actions.

        Parameters:
        -----------
        particle : CellularTypeData
            The cellular type data of the particle to decode.
        others : List[CellularTypeData]
            List of other cellular type data for interactions.
        env : SimulationConfig
            Environmental configuration parameters.
        """
        # Iterate through each gene in the gene sequence
        for gene in self.gene_sequence:
            if not gene or len(gene) < 2:
                continue  # Skip invalid genes
                
            gene_type = gene[0]
            gene_data = gene[1:]
            
            try:
                if gene_type == "start_movement":
                    self.apply_movement_gene(particle, gene_data, env)
                elif gene_type == "start_interaction":
                    self.apply_interaction_gene(particle, others, gene_data, env)
                elif gene_type == "start_energy":
                    self.apply_energy_gene(particle, gene_data, env)
                elif gene_type == "start_reproduction":
                    self.apply_reproduction_gene(particle, others, gene_data, env)
                elif gene_type == "start_growth":
                    self.apply_growth_gene(particle, gene_data)
                elif gene_type == "start_predation":
                    self.apply_predation_gene(particle, others, gene_data, env)
            except Exception as e:
                print(f"Error processing gene {gene_type}: {str(e)}")

    def apply_movement_gene(self, particle: CellularTypeData, gene_data: List[Any], env: SimulationConfig) -> None:
        """
        Apply genes controlling movement behavior.
        gene_data: [speed_modifier, randomness, direction_bias]

        Parameters:
        -----------
        particle : CellularTypeData
            The cellular type data of the particle.
        gene_data : List[Any]
            Data for the movement gene.
        env : SimulationConfig
            Environmental configuration parameters.
        """
        # Default values if gene_data is incomplete
        speed_modifier = gene_data[0] if len(gene_data) > 0 else 1.0
        randomness = gene_data[1] if len(gene_data) > 1 else 0.1
        direction_bias = gene_data[2] if len(gene_data) > 2 else 0.0

        # Clamp values to reasonable ranges
        speed_modifier = np.clip(speed_modifier, 0.1, 3.0)
        randomness = np.clip(randomness, 0.0, 1.0)
        direction_bias = np.clip(direction_bias, -1.0, 1.0)

        # Apply movement modifications vectorized
        friction_factor = 1.0 - env.friction
        particle.vx = particle.vx * friction_factor * speed_modifier + \
                     randomness * np.random.uniform(-1, 1, size=particle.vx.size) + \
                     direction_bias
        particle.vy = particle.vy * friction_factor * speed_modifier + \
                     randomness * np.random.uniform(-1, 1, size=particle.vy.size) + \
                     direction_bias

        # Apply energy cost for movement
        energy_cost = np.sqrt(particle.vx**2 + particle.vy**2) * 0.01
        particle.energy = np.maximum(0.0, particle.energy - energy_cost)

    def apply_interaction_gene(self, particle: CellularTypeData, others: List[CellularTypeData], gene_data: List[Any], env: SimulationConfig) -> None:
        """
        Apply interaction-related behavior based on proximity.
        gene_data: [attraction_strength, interaction_radius]

        Parameters:
        -----------
        particle : CellularTypeData
            The cellular type data of the particle.
        others : List[CellularTypeData]
            List of other cellular type data for interactions.
        gene_data : List[Any]
            Data for the interaction gene.
        env : SimulationConfig
            Environmental configuration parameters.
        """
        # Default values if gene_data is incomplete
        attraction_strength = gene_data[0] if len(gene_data) > 0 else 0.5
        interaction_radius = gene_data[1] if len(gene_data) > 1 else 100.0

        # Clamp values to reasonable ranges
        attraction_strength = np.clip(attraction_strength, -2.0, 2.0)
        interaction_radius = np.clip(interaction_radius, 10.0, 300.0)

        for other in others:
            if other == particle:
                continue

            # Calculate distances and angles vectorized
            dx = other.x - particle.x[:, np.newaxis]
            dy = other.y - particle.y[:, np.newaxis]
            distances = np.sqrt(dx**2 + dy**2)
            
            # Create interaction mask
            interact_mask = (distances > 0.0) & (distances < interaction_radius)
            
            if not np.any(interact_mask):
                continue

            # Calculate normalized direction vectors
            with np.errstate(divide='ignore', invalid='ignore'):
                dx_norm = np.where(distances > 0, dx / distances, 0)
                dy_norm = np.where(distances > 0, dy / distances, 0)

            # Calculate force magnitudes with distance falloff
            force_magnitudes = attraction_strength * (1.0 - distances / interaction_radius)
            
            # Apply forces
            particle.vx += np.sum(dx_norm * force_magnitudes * interact_mask, axis=1)
            particle.vy += np.sum(dy_norm * force_magnitudes * interact_mask, axis=1)

            # Apply small energy cost for interaction
            particle.energy -= 0.01 * np.sum(interact_mask, axis=1)
            particle.energy = np.maximum(0.0, particle.energy)

    def apply_energy_gene(self, particle: CellularTypeData, gene_data: List[Any], env: SimulationConfig) -> None:
        """
        Regulates energy dynamics based on genetic and environmental modifiers.
        gene_data: [passive_gain, feeding_efficiency, predation_efficiency]

        Parameters:
        -----------
        particle : CellularTypeData
            The cellular type data of the particle.
        gene_data : List[Any]
            Data for the energy gene.
        env : SimulationConfig
            Environmental configuration parameters.
        """
        # Default values if gene_data is incomplete
        passive_gain = gene_data[0] if len(gene_data) > 0 else 0.1
        feeding_efficiency = gene_data[1] if len(gene_data) > 1 else 0.5
        predation_efficiency = gene_data[2] if len(gene_data) > 2 else 0.3

        # Clamp values to reasonable ranges
        passive_gain = np.clip(passive_gain, 0.0, 0.5)
        feeding_efficiency = np.clip(feeding_efficiency, 0.1, 1.0)
        predation_efficiency = np.clip(predation_efficiency, 0.1, 1.0)

        # Calculate base energy gain modified by efficiency traits
        base_gain = passive_gain * particle.energy_efficiency

        # Apply environmental modifiers (e.g., day/night cycle, temperature)
        env_modifier = 1.0  # Could be modified based on env parameters
        
        # Calculate total energy gain
        energy_gain = base_gain * env_modifier * feeding_efficiency

        # Apply energy changes vectorized
        particle.energy += energy_gain
        
        # Apply energy decay based on age
        age_factor = np.clip(particle.age / particle.max_age, 0.0, 1.0)
        energy_decay = 0.01 * age_factor
        particle.energy = np.maximum(0.0, particle.energy - energy_decay)

        # Clamp energy to maximum value
        particle.energy = np.minimum(particle.energy, 200.0)

    def apply_reproduction_gene(self, particle: CellularTypeData, others: List[CellularTypeData], gene_data: List[Any], env: SimulationConfig) -> None:
        """
        Handles sexual and asexual reproduction.
        gene_data: [sexual_threshold, asexual_threshold, reproduction_cost, cooldown_time]

        Parameters:
        -----------
        particle : CellularTypeData
            The cellular type data of the particle.
        others : List[CellularTypeData]
            List of other cellular type data for interactions.
        gene_data : List[Any]
            Data for the reproduction gene.
        env : SimulationConfig
            Environmental configuration parameters.
        """
        # Default values if gene_data is incomplete
        sexual_threshold = gene_data[0] if len(gene_data) > 0 else 150.0
        asexual_threshold = gene_data[1] if len(gene_data) > 1 else 100.0
        reproduction_cost = gene_data[2] if len(gene_data) > 2 else 50.0
        cooldown_time = gene_data[3] if len(gene_data) > 3 else 30.0

        # Clamp values to reasonable ranges
        sexual_threshold = np.clip(sexual_threshold, 100.0, 200.0)
        asexual_threshold = np.clip(asexual_threshold, 50.0, 150.0)
        reproduction_cost = np.clip(reproduction_cost, 25.0, 100.0)
        cooldown_time = np.clip(cooldown_time, 10.0, 100.0)

        # Check reproduction conditions
        can_reproduce = (particle.energy > asexual_threshold) & \
                       (particle.age > cooldown_time) & \
                       particle.alive

        if not np.any(can_reproduce):
            return

        # Get indices of particles that can reproduce
        reproduce_indices = np.where(can_reproduce)[0]

        for idx in reproduce_indices:
            # Deduct energy cost
            particle.energy[idx] -= reproduction_cost

            # Create offspring with inherited traits and mutations
            mutation_rate = env.genetics.gene_mutation_rate
            mutation_range = env.genetics.gene_mutation_range

            # Apply mutations to all traits
            offspring_traits = {
                'energy_efficiency': self._mutate_trait(particle.energy_efficiency[idx], mutation_rate, mutation_range),
                'speed_factor': self._mutate_trait(particle.speed_factor[idx], mutation_rate, mutation_range),
                'interaction_strength': self._mutate_trait(particle.interaction_strength[idx], mutation_rate, mutation_range),
                'perception_range': self._mutate_trait(particle.perception_range[idx], mutation_rate, mutation_range),
                'reproduction_rate': self._mutate_trait(particle.reproduction_rate[idx], mutation_rate, mutation_range),
                'synergy_affinity': self._mutate_trait(particle.synergy_affinity[idx], mutation_rate, mutation_range),
                'colony_factor': self._mutate_trait(particle.colony_factor[idx], mutation_rate, mutation_range),
                'drift_sensitivity': self._mutate_trait(particle.drift_sensitivity[idx], mutation_rate, mutation_range)
            }

            # Calculate genetic distance for speciation
            genetic_distance = np.sqrt(
                (offspring_traits['speed_factor'] - particle.speed_factor[idx])**2 +
                (offspring_traits['interaction_strength'] - particle.interaction_strength[idx])**2 +
                (offspring_traits['perception_range'] - particle.perception_range[idx])**2 +
                (offspring_traits['reproduction_rate'] - particle.reproduction_rate[idx])**2 +
                (offspring_traits['synergy_affinity'] - particle.synergy_affinity[idx])**2 +
                (offspring_traits['colony_factor'] - particle.colony_factor[idx])**2 +
                (offspring_traits['drift_sensitivity'] - particle.drift_sensitivity[idx])**2
            )

            # Determine species ID
            if genetic_distance > env.speciation_threshold:
                species_id_val = int(np.max(particle.species_id)) + 1
            else:
                species_id_val = particle.species_id[idx]

            # Add offspring to particle data
            particle.add_component(
                x=particle.x[idx] + np.random.uniform(-5, 5),
                y=particle.y[idx] + np.random.uniform(-5, 5),
                vx=particle.vx[idx] * np.random.uniform(0.9, 1.1),
                vy=particle.vy[idx] * np.random.uniform(0.9, 1.1),
                energy=particle.energy[idx] * 0.5,
                mass_val=particle.mass[idx] if particle.mass_based else None,
                energy_efficiency_val=offspring_traits['energy_efficiency'],
                speed_factor_val=offspring_traits['speed_factor'],
                interaction_strength_val=offspring_traits['interaction_strength'],
                perception_range_val=offspring_traits['perception_range'],
                reproduction_rate_val=offspring_traits['reproduction_rate'],
                synergy_affinity_val=offspring_traits['synergy_affinity'],
                colony_factor_val=offspring_traits['colony_factor'],
                drift_sensitivity_val=offspring_traits['drift_sensitivity'],
                species_id_val=species_id_val,
                parent_id_val=particle.type_id,
                max_age=particle.max_age
            )

    def _mutate_trait(self, base_value: float, mutation_rate: float, mutation_range: Tuple[float, float]) -> float:
        """Helper method to mutate a trait value with given parameters."""
        if np.random.random() < mutation_rate:
            mutation = np.random.uniform(mutation_range[0], mutation_range[1])
            return np.clip(base_value + mutation, 0.1, 3.0)
        return base_value

    def apply_growth_gene(self, particle: CellularTypeData, gene_data: List[Any]) -> None:
        """
        Handles juvenile-to-adult growth and sexual maturity.
        gene_data: [growth_rate, adult_size, maturity_age]

        Parameters:
        -----------
        particle : CellularTypeData
            The cellular type data of the particle.
        gene_data : List[Any]
            Data for the growth gene.
        """
        # Default values if gene_data is incomplete
        growth_rate = gene_data[0] if len(gene_data) > 0 else 0.1
        adult_size = gene_data[1] if len(gene_data) > 1 else 2.0
        maturity_age = gene_data[2] if len(gene_data) > 2 else 100.0

        # Clamp values to reasonable ranges
        growth_rate = np.clip(growth_rate, 0.01, 0.5)
        adult_size = np.clip(adult_size, 1.0, 5.0)
        maturity_age = np.clip(maturity_age, 50.0, 200.0)

        # Calculate growth factor based on age
        juvenile_mask = particle.age < maturity_age
        growth_factor = np.where(juvenile_mask,
                               growth_rate * (1.0 - particle.age / maturity_age),
                               0.0)

        # Apply growth effects
        particle.energy += growth_factor * particle.energy_efficiency
        
        if particle.mass_based and particle.mass is not None:
            # Grow mass for juvenile particles
            particle.mass = np.where(juvenile_mask,
                                   particle.mass * (1.0 + growth_factor),
                                   particle.mass)
            particle.mass = np.clip(particle.mass, 0.1, adult_size)

    def apply_predation_gene(self, particle: CellularTypeData, others: List[CellularTypeData], gene_data: List[Any], env: SimulationConfig) -> None:
        """
        Encodes predatory behavior for attacking and feeding on other particles.
        gene_data: [attack_power, energy_gain]

        Parameters:
        -----------
        particle : CellularTypeData
            The cellular type data of the particle.
        others : List[CellularTypeData]
            List of other cellular type data for interactions.
        gene_data : List[Any]
            Data for the predation gene.
        env : SimulationConfig
            Environmental configuration parameters.
        """
        # Default values if gene_data is incomplete
        attack_power = gene_data[0] if len(gene_data) > 0 else 10.0
        energy_gain = gene_data[1] if len(gene_data) > 1 else 5.0

        # Clamp values to reasonable ranges
        attack_power = np.clip(attack_power, 1.0, 20.0)
        energy_gain = np.clip(energy_gain, 1.0, 10.0)

        for other in others:
            if other == particle:
                continue

            # Calculate distances vectorized
            dx = other.x - particle.x[:, np.newaxis]
            dy = other.y - particle.y[:, np.newaxis]
            distances = np.sqrt(dx**2 + dy**2)

            # Create predation mask
            predation_mask = (distances < env.predation_range) & \
                           other.alive[np.newaxis, :] & \
                           (particle.energy[:, np.newaxis] > other.energy)

            if not np.any(predation_mask):
                continue

            # Get predator-prey pairs
            pred_idx, prey_idx = np.where(predation_mask)

            # Calculate damage based on attack power and relative energy levels
            energy_ratio = particle.energy[pred_idx] / other.energy[prey_idx]
            damage = attack_power * energy_ratio

            # Apply damage to prey
            other.energy[prey_idx] -= damage

            # Predators gain energy
            gained_energy = energy_gain * damage * particle.energy_efficiency[pred_idx]
            particle.energy[pred_idx] += gained_energy

            # Update prey alive status
            other.alive[prey_idx] = other.energy[prey_idx] > 0

            # Clamp energies to valid range
            particle.energy = np.clip(particle.energy, 0.0, 200.0)
            other.energy = np.clip(other.energy, 0.0, 200.0)

###############################################################
# Interaction Rules, Give-Take & Synergy
###############################################################

class InteractionRules:
    """
    Manages creation and evolution of interaction parameters, give-take matrix, and synergy matrix.
    """
    
    def __init__(self, config: SimulationConfig, mass_based_type_indices: List[int]):
        """
        Initialize the InteractionRules with given configuration and mass-based cellular type indices.

        Parameters:
        -----------
        config : SimulationConfig
            Configuration parameters for the simulation.
        mass_based_type_indices : List[int]
            List of cellular type indices that are mass-based.
        """
        self.config = config  # Store simulation configuration
        self.mass_based_type_indices = mass_based_type_indices  # Indices of mass-based cellular types
        # Create initial interaction rules between cellular type pairs
        self.rules = self._create_interaction_matrix()
        # Create give-take relationships between cellular types
        self.give_take_matrix = self._create_give_take_matrix()
        # Create synergy (alliance) relationships between cellular types
        self.synergy_matrix = self._create_synergy_matrix()

    def _create_interaction_matrix(self) -> List[Tuple[int, int, Dict[str, Any]]]:
        """
        Create a static NxN interaction matrix with parameters for each cellular type pair.

        Returns:
        --------
        List[Tuple[int, int, Dict[str, Any]]]
            List of tuples containing cellular type indices and their interaction parameters.
        """
        final_rules = []  # List to hold interaction rules
        for i in range(self.config.n_cell_types):
            for j in range(self.config.n_cell_types):
                # Generate interaction parameters between cellular type i and j
                params = self._random_interaction_params(i, j)
                final_rules.append((i, j, params))  # Append to rules list
        return final_rules  # Return the complete interaction matrix

    def _random_interaction_params(self, i: int, j: int) -> Dict[str, Any]:
        """
        Generate random interaction parameters between cellular type i and j.

        Parameters:
        -----------
        i : int
            Index of the first cellular type.
        j : int
            Index of the second cellular type.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing interaction parameters.
        """
        # Determine if both cellular types are mass-based
        both_mass = (i in self.mass_based_type_indices and j in self.mass_based_type_indices)
        # Decide randomly whether to use gravity if both are mass-based
        use_gravity = both_mass and (random.random() < 0.5)
        use_potential = True  # Always use potential-based interactions

        # Randomly assign potential strength within the specified range
        potential_strength = random.uniform(self.config.interaction_strength_range[0],
                                           self.config.interaction_strength_range[1])
        if random.random() < 0.5:
            potential_strength = -potential_strength  # Randomly invert potential strength

        # Assign gravity factor if gravity is used
        gravity_factor = random.uniform(0.1, 2.0) if use_gravity else 0.0
        # Assign maximum interaction distance
        max_dist = random.uniform(50.0, 200.0)

        # Compile interaction parameters into a dictionary
        params = {
            "use_potential": use_potential,  # Whether to use potential-based forces
            "use_gravity": use_gravity,      # Whether to use gravity-based forces
            "potential_strength": potential_strength,  # Strength of potential
            "gravity_factor": gravity_factor,          # Gravity factor if applicable
            "max_dist": max_dist                         # Maximum interaction distance
        }
        return params  # Return the interaction parameters

    def _create_give_take_matrix(self) -> np.ndarray:
        """
        Create a give-take matrix indicating which cellular types give energy to which.

        Returns:
        --------
        np.ndarray
            NxN boolean matrix where element (i,j) is True if cellular type i gives energy to cellular type j.
        """
        matrix = np.zeros((self.config.n_cell_types, self.config.n_cell_types), dtype=bool)  # Initialize matrix
        for i in range(self.config.n_cell_types):
            for j in range(self.config.n_cell_types):
                if i != j:
                    # 10% chance that cellular type i gives energy to cellular type j
                    if random.random() < 0.1:
                        matrix[i, j] = True
        return matrix  # Return the give-take matrix

    def _create_synergy_matrix(self) -> np.ndarray:
        """
        Create a synergy matrix indicating energy sharing between cellular types.

        Returns:
        --------
        np.ndarray
            NxN float matrix where element (i,j) > 0 indicates synergy factor between cellular type i and j.
        """
        synergy_matrix = np.zeros((self.config.n_cell_types, self.config.n_cell_types), dtype=float)  # Initialize matrix
        for i in range(self.config.n_cell_types):
            for j in range(self.config.n_cell_types):
                if i != j:
                    if random.random() < 0.1:
                        # 10% chance to establish synergy with a random factor between 0.01 and 0.3
                        synergy_matrix[i, j] = random.uniform(0.01, 0.3)
                    else:
                        synergy_matrix[i, j] = 0.0  # No synergy
        return synergy_matrix  # Return the synergy matrix

    def evolve_parameters(self, frame_count: int) -> None:
        """
        Evolve interaction parameters, give-take parameters, and synergy periodically.

        Parameters:
        -----------
        frame_count : int
            Current frame count of the simulation.
        """
        if frame_count % self.config.evolution_interval == 0:
            # Evolve interaction rules
            for _, _, params in self.rules:
                if random.random() < 0.1:
                    # 10% chance to slightly mutate potential_strength
                    params["potential_strength"] *= random.uniform(0.95, 1.05)
                if random.random() < 0.05 and "gravity_factor" in params:
                    # 5% chance to slightly mutate gravity_factor
                    params["gravity_factor"] *= random.uniform(0.95, 1.05)
                if random.random() < 0.05:
                    # 5% chance to slightly mutate max_dist, ensuring it doesn't go below 10
                    params["max_dist"] = max(10.0, params["max_dist"] * random.uniform(0.95, 1.05))

            # Evolve give-take energy transfer factor with a 10% chance
            if random.random() < 0.1:
                self.config.energy_transfer_factor = min(
                    1.0,  # Ensure it does not exceed 1.0
                    self.config.energy_transfer_factor * random.uniform(0.95, 1.05)
                )

            # Evolve synergy matrix with a 5% chance per element
            for i in range(self.synergy_matrix.shape[0]):
                for j in range(self.synergy_matrix.shape[1]):
                    if random.random() < 0.05:
                        # Slightly adjust synergy factor, keeping it within [0.0, 1.0]
                        self.synergy_matrix[i, j] = min(
                            1.0,
                            max(0.0, self.synergy_matrix[i, j] + (random.random() * 0.1 - 0.05))
                        )

###############################################################
# Cellular Type Manager (Handles Multi-Type Operations & Reproduction)
###############################################################

class CellularTypeManager:
    """
    Manages all cellular types in the simulation. Handles reproduction, interaction between types, and updates.
    """
    
    def __init__(self, config: SimulationConfig, colors: List[Tuple[int, int, int]], mass_based_type_indices: List[int]):
        """
        Initialize the CellularTypeManager with configuration, colors, and mass-based type indices.

        Parameters:
        -----------
        config : SimulationConfig
            Configuration parameters for the simulation.
        colors : List[Tuple[int, int, int]]
            List of RGB color tuples for each cellular type.
        mass_based_type_indices : List[int]
            List of cellular type indices that are mass-based.
        """
        self.config = config  # Store simulation configuration
        self.cellular_types: List[CellularTypeData] = []  # List to hold all cellular type data
        self.mass_based_type_indices = mass_based_type_indices  # Indices of mass-based types
        self.colors = colors  # Colors assigned to cellular types

    def add_cellular_type_data(self, data: CellularTypeData) -> None:
        """
        Add a CellularTypeData instance to the manager.

        Parameters:
        -----------
        data : CellularTypeData
            The cellular type data to add.
        """
        self.cellular_types.append(data)  # Append cellular type data to the list

    def get_cellular_type_by_id(self, i: int) -> CellularTypeData:
        """
        Retrieve cellular type data by its ID.

        Parameters:
        -----------
        i : int
            Cellular type ID.

        Returns:
        --------
        CellularTypeData
            The cellular type data corresponding to the given ID.
        """
        return self.cellular_types[i]  # Return cellular type data at index i

    def remove_dead_in_all_types(self) -> None:
        """
        Remove dead cellular components from all types managed by the manager.
        """
        for ct in self.cellular_types:
            ct.remove_dead(self.config)  # Remove dead components from each type with access to config

    def reproduce(self) -> None:
        """
        Attempt to reproduce cellular components in each type if conditions are met.
        Offspring inherit properties with possible mutations based on genetic parameters.
        This method is fully vectorized for performance and handles all genetic traits.
        """
        for ct in self.cellular_types:
            # Early exit if at max capacity
            if ct.x.size >= self.config.max_particles_per_type:
                continue

            # Ensure arrays are same size before boolean indexing
            if ct.reproduction_rate.size != ct.x.size:
                continue

            # Identify eligible components for reproduction
            eligible = (ct.alive & 
                      (ct.energy > self.config.reproduction_energy_threshold) &
                      (np.random.random(ct.x.size) < ct.reproduction_rate))
            
            num_offspring = np.sum(eligible)
            if num_offspring == 0:
                continue

            # Get parent indices and validate
            parent_indices = np.where(eligible)[0]
            if parent_indices.size == 0:
                continue

            # Parents lose half their energy
            ct.energy[eligible] *= 0.5
            parent_energy = ct.energy[eligible]

            # Calculate offspring energy
            offspring_energy = parent_energy * self.config.reproduction_offspring_energy_fraction

            # Create mutation mask
            mutation_mask = np.random.random(num_offspring) < self.config.genetics.gene_mutation_rate

            # Inherit and mutate genetic traits
            offspring_traits = {}
            for trait in self.config.genetics.gene_traits:
                parent_values = getattr(ct, trait)[parent_indices]
                offspring_traits[trait] = parent_values.copy()
                
                if mutation_mask.any():
                    mutation = np.random.uniform(
                        self.config.genetics.gene_mutation_range[0],
                        self.config.genetics.gene_mutation_range[1],
                        size=mutation_mask.sum()
                    )
                    offspring_traits[trait][mutation_mask] += mutation

            # Clamp genetic values
            (offspring_traits["speed_factor"],
             offspring_traits["interaction_strength"],
             offspring_traits["perception_range"],
             offspring_traits["reproduction_rate"],
             offspring_traits["synergy_affinity"],
             offspring_traits["colony_factor"],
             offspring_traits["drift_sensitivity"]) = self.config.genetics.clamp_gene_values(
                offspring_traits["speed_factor"],
                offspring_traits["interaction_strength"], 
                offspring_traits["perception_range"],
                offspring_traits["reproduction_rate"],
                offspring_traits["synergy_affinity"],
                offspring_traits["colony_factor"],
                offspring_traits["drift_sensitivity"]
            )

            # Handle energy efficiency
            offspring_efficiency = ct.energy_efficiency[parent_indices].copy()
            if mutation_mask.any():
                efficiency_mutation = np.random.uniform(
                    self.config.genetics.energy_efficiency_mutation_range[0],
                    self.config.genetics.energy_efficiency_mutation_range[1],
                    size=mutation_mask.sum()
                )
                offspring_efficiency[mutation_mask] += efficiency_mutation
            offspring_efficiency = np.clip(
                offspring_efficiency,
                self.config.energy_efficiency_range[0],
                self.config.energy_efficiency_range[1]
            )

            # Handle mass for mass-based types
            offspring_mass = None
            if ct.mass_based and ct.mass is not None:
                offspring_mass = ct.mass[parent_indices].copy()
                if mutation_mask.any():
                    offspring_mass[mutation_mask] *= np.random.uniform(0.95, 1.05, size=mutation_mask.sum())
                offspring_mass = np.maximum(offspring_mass, 0.1)

            # Calculate genetic distance and species IDs
            genetic_distance = np.sqrt(np.sum([
                (offspring_traits[trait] - getattr(ct, trait)[parent_indices]) ** 2 
                for trait in self.config.genetics.gene_traits
            ], axis=0))

            max_species_id = np.max(ct.species_id) if ct.species_id.size > 0 else 0
            new_species_ids = np.where(
                genetic_distance > self.config.speciation_threshold,
                max_species_id + 1,
                ct.species_id[parent_indices]
            )

            # Add offspring components safely
            for i in range(num_offspring):
                try:
                    ct.add_component(
                        x=ct.x[parent_indices[i]],
                        y=ct.y[parent_indices[i]],
                        vx=np.random.uniform(-0.5, 0.5) * self.config.base_velocity_scale / offspring_efficiency[i] * offspring_traits["speed_factor"][i],
                        vy=np.random.uniform(-0.5, 0.5) * self.config.base_velocity_scale / offspring_efficiency[i] * offspring_traits["speed_factor"][i],
                        energy=offspring_energy[i],
                        mass_val=offspring_mass[i] if offspring_mass is not None else None,
                        energy_efficiency_val=offspring_efficiency[i],
                        speed_factor_val=offspring_traits["speed_factor"][i],
                        interaction_strength_val=offspring_traits["interaction_strength"][i],
                        perception_range_val=offspring_traits["perception_range"][i],
                        reproduction_rate_val=offspring_traits["reproduction_rate"][i],
                        synergy_affinity_val=offspring_traits["synergy_affinity"][i],
                        colony_factor_val=offspring_traits["colony_factor"][i],
                        drift_sensitivity_val=offspring_traits["drift_sensitivity"][i],
                        species_id_val=new_species_ids[i],
                        parent_id_val=ct.type_id,
                        max_age=ct.max_age
                    )
                except IndexError:
                    continue  # Skip if index out of bounds

###############################################################
# Forces & Interactions
###############################################################

def apply_interaction(a_x: float, a_y: float, b_x: float, b_y: float, params: Dict[str, Any]) -> Tuple[float, float]:
    """
    Compute force exerted by cellular component B on A given interaction parameters.
    This is a per-pair calculation.

    Parameters:
    -----------
    a_x : float
        X-coordinate of cellular component A.
    a_y : float
        Y-coordinate of cellular component A.
    b_x : float
        X-coordinate of cellular component B.
    b_y : float
        Y-coordinate of cellular component B.
    params : Dict[str, Any]
        Dictionary containing interaction parameters.

    Returns:
    --------
    Tuple[float, float]
        Force components (fx, fy) exerted on cellular component A by cellular component B.
    """
    dx = a_x - b_x  # Difference in x-coordinate
    dy = a_y - b_y  # Difference in y-coordinate
    d_sq = dx * dx + dy * dy  # Squared distance between components

    if d_sq == 0.0 or d_sq > params["max_dist"] ** 2:
        return 0.0, 0.0  # No force if components overlap or are beyond max_dist

    d = math.sqrt(d_sq)  # Euclidean distance between components

    fx, fy = 0.0, 0.0  # Initialize force components

    # Potential-based interaction
    if params.get("use_potential", True):
        pot_strength = params.get("potential_strength", 1.0)  # Strength of potential
        F_pot = pot_strength / d  # Force magnitude inversely proportional to distance
        fx += F_pot * dx  # X-component of potential force
        fy += F_pot * dy  # Y-component of potential force

    # Gravity-based interaction
    if params.get("use_gravity", False):
        # Gravity factor requires masses of both components, assumed to be provided in params
        if "m_a" in params and "m_b" in params:
            m_a = params["m_a"]  # Mass of component A
            m_b = params["m_b"]  # Mass of component B
            gravity_factor = params.get("gravity_factor", 1.0)  # Gravity scaling factor
            F_grav = gravity_factor * (m_a * m_b) / d_sq  # Gravitational force magnitude
            fx += F_grav * dx  # X-component of gravitational force
            fy += F_grav * dy  # Y-component of gravitational force

    return fx, fy  # Return total force components

def give_take_interaction(giver_energy: float, receiver_energy: float,
                          giver_mass: Optional[float], receiver_mass: Optional[float],
                          config: SimulationConfig) -> Tuple[float, float, Optional[float], Optional[float]]:
    """
    Transfer energy (and mass if enabled) from giver to receiver.

    Parameters:
    -----------
    giver_energy : float
        Current energy level of the giver.
    receiver_energy : float
        Current energy level of the receiver.
    giver_mass : Optional[float]
        Current mass of the giver (if mass-based).
    receiver_mass : Optional[float]
        Current mass of the receiver (if mass-based).
    config : SimulationConfig
        Configuration parameters for the simulation.

    Returns:
    --------
    Tuple[float, float, Optional[float], Optional[float]]
        Updated (giver_energy, receiver_energy, giver_mass, receiver_mass).
    """
    transfer_amount = receiver_energy * config.energy_transfer_factor  # Calculate energy to transfer
    receiver_energy -= transfer_amount  # Decrease receiver's energy
    giver_energy += transfer_amount  # Increase giver's energy

    if config.mass_transfer and receiver_mass is not None and giver_mass is not None:
        mass_transfer_amount = receiver_mass * config.energy_transfer_factor  # Calculate mass to transfer
        receiver_mass -= mass_transfer_amount  # Decrease receiver's mass
        giver_mass += mass_transfer_amount  # Increase giver's mass

    return giver_energy, receiver_energy, giver_mass, receiver_mass  # Return updated values

###############################################################
# Clustering & Synergy Functions
###############################################################

def apply_synergy(energyA: float, energyB: float, synergy_factor: float) -> Tuple[float, float]:
    """
    Apply synergy: share energy between two allied cellular type components.

    Parameters:
    -----------
    energyA : float
        Current energy of component A.
    energyB : float
        Current energy of component B.
    synergy_factor : float
        Factor determining how much energy to share.

    Returns:
    --------
    Tuple[float, float]
        Updated energies of component A and component B after synergy.
    """
    avg_energy = (energyA + energyB) * 0.5  # Calculate average energy
    newA = (energyA * (1.0 - synergy_factor)) + (avg_energy * synergy_factor)  # Update energy of A
    newB = (energyB * (1.0 - synergy_factor)) + (avg_energy * synergy_factor)  # Update energy of B
    return newA, newB  # Return updated energies

###############################################################
# Renderer Class
###############################################################

class Renderer:
    """
    Handles rendering of cellular components on the Pygame window.
    """
    
    def __init__(self, surface: pygame.Surface, config: SimulationConfig):
        """
        Initialize the Renderer with a Pygame surface and configuration.

        Parameters:
        -----------
        surface : pygame.Surface
            The Pygame surface where cellular components will be drawn.
        config : SimulationConfig
            Configuration parameters for the simulation.
        """
        self.surface = surface  # Pygame surface for rendering
        self.config = config    # Simulation configuration

        # Precompute a surface for all particles to optimize rendering
        self.particle_surface = pygame.Surface(self.surface.get_size(), flags=pygame.SRCALPHA)
        self.particle_surface = self.particle_surface.convert_alpha()

        # Initialize font for rendering statistics
        pygame.font.init()  # Initialize Pygame font module
        self.font = pygame.font.SysFont('Arial', 20)  # Use Arial font, size 20

    def draw_component(self, x: float, y: float, color: Tuple[int, int, int], energy: float, speed_factor: float) -> None:
        """
        Draw a single cellular component as a circle on the surface. Energy is mapped to brightness.

        Parameters:
        -----------
        x : float
            X-coordinate of the component.
        y : float
            Y-coordinate of the component.
        color : Tuple[int, int, int]
            RGB color tuple of the component.
        energy : float
            Current energy level of the component, influencing brightness.
        speed_factor : float
            Speed factor gene trait influencing the brightness scaling.
        """
        # Clamp energy between 0 and 100 for brightness scaling
        health = min(100.0, max(0.0, energy))
        # Normalize energy to [0,1] for intensity factor
        intensity_factor = max(0.0, min(1.0, health / 100.0))

        # Adjust color brightness based on energy and speed_factor
        # Higher energy and speed_factor -> brighter color; lower values -> dimmer color with a base minimum brightness
        c = (
            min(255, int(color[0] * intensity_factor * speed_factor + (1 - intensity_factor) * 100)),
            min(255, int(color[1] * intensity_factor * speed_factor + (1 - intensity_factor) * 100)),
            min(255, int(color[2] * intensity_factor * speed_factor + (1 - intensity_factor) * 100))
        )

        # Draw the component as a filled circle with the calculated color and size
        pygame.draw.circle(self.particle_surface, c, (int(x), int(y)), int(self.config.particle_size))

    def draw_cellular_type(self, ct: CellularTypeData) -> None:
        """
        Draw all alive cellular components of a cellular type on the surface.

        Parameters:
        -----------
        ct : CellularTypeData
            The cellular type data containing components to draw.
        """
        # Utilize NumPy's vectorization to efficiently iterate over alive components
        alive_indices = np.where(ct.alive)[0]  # Get indices of alive components
        # Iterate over alive components and draw them
        for idx in alive_indices:
            self.draw_component(ct.x[idx], ct.y[idx], ct.color, ct.energy[idx], ct.speed_factor[idx])

    def render(self, stats: Dict[str, Any]) -> None:
        """
        Blit the particle surface onto the main surface and reset the particle surface for the next frame.
        Additionally, render simulation statistics on the screen.

        Parameters:
        -----------
        stats : Dict[str, Any]
            Dictionary containing simulation statistics to display.
        """
        self.surface.blit(self.particle_surface, (0, 0))  # Blit all particles at once
        self.particle_surface.fill((0, 0, 0, 0))  # Clear particle surface for next frame

        # Render statistics on the top-left corner
        stats_text = f"FPS: {stats.get('fps', 0):.2f} | Total Species: {stats.get('total_species', 0)} | Total Particles: {stats.get('total_particles', 0)}"
        text_surface = self.font.render(stats_text, True, (255, 255, 255))  # Render text in white
        self.surface.blit(text_surface, (10, 10))  # Position text at (10,10)

###############################################################
# Cellular Automata (Main Simulation)
###############################################################

class CellularAutomata:
    """
    The main simulation class. Initializes and runs the simulation loop.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the CellularAutomata with the given configuration.

        Parameters:
        -----------
        config : SimulationConfig
            Configuration parameters for the simulation.
        """
        self.config = config  # Store simulation configuration
        pygame.init()  # Initialize all imported Pygame modules

        # Retrieve display information to set fullscreen window
        display_info = pygame.display.Info()
        screen_width, screen_height = display_info.current_w, display_info.current_h  # Current display dimensions

        # Set up a fullscreen window with the calculated dimensions
        self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)  # Create optimized fullscreen window
        pygame.display.set_caption("Emergent Cellular Automata Simulation")  # Set window title

        self.clock = pygame.time.Clock()  # Clock to manage frame rate
        self.frame_count = 0  # Initialize frame counter
        self.run_flag = True  # Flag to control main loop

        # Calculate minimum distance from edges (5% of the larger screen dimension)
        self.edge_buffer = 0.05 * max(screen_width, screen_height)  # Minimum distance from edges in pixels

        # Setup cellular type colors and identify mass-based types
        self.colors = generate_vibrant_colors(self.config.n_cell_types)  # Generate colors for cellular types
        n_mass_types = int(self.config.mass_based_fraction * self.config.n_cell_types)  # Number of mass-based types
        mass_based_type_indices = list(range(n_mass_types))  # Indices of mass-based types

        # Initialize CellularTypeManager with configuration, colors, and mass-based type indices
        self.type_manager = CellularTypeManager(self.config, self.colors, mass_based_type_indices)

        # Pre-calculate mass values for efficiency
        mass_values = np.random.uniform(self.config.mass_range[0], self.config.mass_range[1], n_mass_types)

        # Create cellular type data for each type and add to CellularTypeManager
        for i in range(self.config.n_cell_types):
            ct = CellularTypeData(
                type_id=i,
                color=self.colors[i],
                n_particles=self.config.particles_per_type,
                window_width=screen_width,
                window_height=screen_height,
                initial_energy=self.config.initial_energy,
                max_age=self.config.max_age,
                mass=mass_values[i] if i < n_mass_types else None,
                base_velocity_scale=self.config.base_velocity_scale
            )
            self.type_manager.add_cellular_type_data(ct)  # Add type to manager

        # Initialize InteractionRules with configuration and mass-based type indices
        self.rules_manager = InteractionRules(self.config, mass_based_type_indices)

        # Initialize Renderer with Pygame surface and configuration
        self.renderer = Renderer(self.screen, self.config)

        # Initialize GeneticInterpreter for decoding genetic traits
        self.genetic_interpreter = GeneticInterpreter()

        # Initialize statistics tracking using NumPy array for better performance
        self.species_count = defaultdict(int)  # Tracks number of species
        self.update_species_count()  # Initialize species count

        # Pre-allocate arrays for boundary calculations
        self.screen_bounds = np.array([
            self.edge_buffer,
            screen_width - self.edge_buffer,
            self.edge_buffer,
            screen_height - self.edge_buffer
        ])

    def update_species_count(self) -> None:
        """
        Update the species count based on current cellular types.
        """
        self.species_count.clear()  # Reset species count
        for ct in self.type_manager.cellular_types:
            unique, counts = np.unique(ct.species_id, return_counts=True)
            for species, count in zip(unique, counts):
                self.species_count[species] += count

    def main_loop(self) -> None:
        """
        Run the main simulation loop until exit conditions are met.
        """
        while self.run_flag:
            self.frame_count += 1  # Increment frame counter
            if self.config.max_frames > 0 and self.frame_count > self.config.max_frames:
                self.run_flag = False  # Exit if max_frames is reached

            # Handle Pygame events (e.g., window close, key presses)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.run_flag = False  # Exit on quit event or ESC key
                    break

            # Evolve interaction parameters periodically
            self.rules_manager.evolve_parameters(self.frame_count)

            # Decode genetic traits and apply gene-based behaviors
            self.decode_genetic_traits()

            # Clear the main screen by filling it with dark gray to prevent piling up
            self.screen.fill((69, 69, 69))  # Fill the main screen with dark gray

            # Apply all inter-type interactions and updates
            self.apply_all_interactions()

            # Apply clustering within each cellular type to simulate flocking behavior
            for ct in self.type_manager.cellular_types:
                self.apply_clustering(ct)

            # Handle reproduction of cellular components across all types
            self.type_manager.reproduce()

            # Remove dead cellular components from all types
            self.type_manager.remove_dead_in_all_types()

            # Update species count after reproduction and death
            self.update_species_count()

            # Draw all cellular types' components on the particle surface
            for ct in self.type_manager.cellular_types:
                self.renderer.draw_cellular_type(ct)

            # Handle boundary reflections after updating positions
            self.handle_boundary_reflections()

            # Compile statistics for rendering
            stats = {
                "fps": self.clock.get_fps(),
                "total_species": len(self.species_count),
                "total_particles": sum(self.species_count.values())
            }

            # Render all particles and statistics onto the main screen
            self.renderer.render(stats)

            # Update the full display surface to the screen
            pygame.display.flip()

            # Cap the frame rate and retrieve the current FPS
            current_fps = self.clock.tick(120)  # Attempt to cap at 120 FPS

            # Adaptive Particle Management - only check every 10th frame
            if current_fps <= 60:
                self.cull_oldest_particles()

        pygame.quit()  # Quit Pygame when the loop ends

    def decode_genetic_traits(self) -> None:
        """
        Decode and apply genetic traits for all cellular types.
        Ensures that all genetic traits influence particle behavior.
        """
        for ct in self.type_manager.cellular_types:
            # Gather all other cellular types for interaction
            others = [other_ct for other_ct in self.type_manager.cellular_types if other_ct != ct]
            # Decode genetic traits to influence behavior
            self.genetic_interpreter.decode(ct, others, self.config)

    def apply_all_interactions(self) -> None:
        """
        Apply inter-type interactions: forces, give-take, and synergy.
        """
        # Iterate over all interaction rules between cellular type pairs
        for (i, j, params) in self.rules_manager.rules:
            self.apply_interaction_between_types(i, j, params)

    def apply_interaction_between_types(self, i: int, j: int, params: Dict[str, Any]) -> None:
        """
        Apply interaction rules between cellular type i and cellular type j.
        This includes forces, give-take, and synergy.

        Parameters:
        -----------
        i : int
            Index of the first cellular type.
        j : int
            Index of the second cellular type.
        params : Dict[str, Any]
            Interaction parameters between cellular type i and j.
        """
        ct_i = self.type_manager.get_cellular_type_by_id(i)  # Get cellular type i
        ct_j = self.type_manager.get_cellular_type_by_id(j)  # Get cellular type j

        # Extract synergy factor and give-take relationship from interaction rules
        synergy_factor = self.rules_manager.synergy_matrix[i, j]  # Synergy factor between types i and j
        is_giver = self.rules_manager.give_take_matrix[i, j]  # Give-take relationship flag

        n_i = ct_i.x.size  # Number of components in type i
        n_j = ct_j.x.size  # Number of components in type j

        if n_i == 0 or n_j == 0:
            return  # No interaction if one type has no components

        # Prepare mass parameters if gravity is used
        if params.get("use_gravity", False):
            if (ct_i.mass_based and ct_i.mass is not None and
                ct_j.mass_based and ct_j.mass is not None):
                params["m_a"] = ct_i.mass  # Mass of type i components
                params["m_b"] = ct_j.mass  # Mass of type j components
            else:
                params["use_gravity"] = False

        # Calculate pairwise differences using broadcasting
        dx = ct_i.x[:, np.newaxis] - ct_j.x  # Shape: (n_i, n_j)
        dy = ct_i.y[:, np.newaxis] - ct_j.y  # Shape: (n_i, n_j)
        dist_sq = dx * dx + dy * dy  # Squared distances, shape: (n_i, n_j)

        # Determine which pairs are within interaction range and not overlapping
        within_range = (dist_sq > 0.0) & (dist_sq <= params["max_dist"] ** 2)  # Boolean mask

        # Extract indices of interacting pairs
        indices = np.where(within_range)
        if len(indices[0]) == 0:
            return  # No interactions within range

        # Calculate distances for interacting pairs
        dist = np.sqrt(dist_sq[indices])

        # Initialize force arrays
        fx = np.zeros_like(dist)
        fy = np.zeros_like(dist)

        # Potential-based forces
        if params.get("use_potential", True):
            pot_strength = params.get("potential_strength", 1.0)
            F_pot = pot_strength / dist
            fx += F_pot * dx[indices]
            fy += F_pot * dy[indices]

        # Gravity-based forces
        if params.get("use_gravity", False):
            gravity_factor = params.get("gravity_factor", 1.0)
            F_grav = gravity_factor * (params["m_a"][indices[0]] * params["m_b"][indices[1]]) / dist_sq[indices]
            fx += F_grav * dx[indices]
            fy += F_grav * dy[indices]

        # Accumulate forces using NumPy's add.at for atomic operations
        np.add.at(ct_i.vx, indices[0], fx)
        np.add.at(ct_i.vy, indices[0], fy)

        # Handle give-take interactions
        if is_giver:
            give_take_within = dist_sq[indices] <= self.config.predation_range ** 2
            give_take_indices = (indices[0][give_take_within], indices[1][give_take_within])
            if give_take_indices[0].size > 0:
                giver_energy = ct_i.energy[give_take_indices[0]]
                receiver_energy = ct_j.energy[give_take_indices[1]]
                giver_mass = ct_i.mass[give_take_indices[0]] if ct_i.mass_based else None
                receiver_mass = ct_j.mass[give_take_indices[1]] if ct_j.mass_based else None

                updated = give_take_interaction(
                    giver_energy,
                    receiver_energy,
                    giver_mass,
                    receiver_mass,
                    self.config
                )
                ct_i.energy[give_take_indices[0]] = updated[0]
                ct_j.energy[give_take_indices[1]] = updated[1]

                if ct_i.mass_based and ct_i.mass is not None and updated[2] is not None:
                    ct_i.mass[give_take_indices[0]] = updated[2]
                if ct_j.mass_based and ct_j.mass is not None and updated[3] is not None:
                    ct_j.mass[give_take_indices[1]] = updated[3]

        # Handle synergy interactions
        if synergy_factor > 0.0 and self.config.synergy_range > 0.0:
            synergy_within = dist_sq[indices] <= self.config.synergy_range ** 2
            synergy_indices = (indices[0][synergy_within], indices[1][synergy_within])
            if synergy_indices[0].size > 0:
                energyA = ct_i.energy[synergy_indices[0]]
                energyB = ct_j.energy[synergy_indices[1]]
                new_energyA, new_energyB = apply_synergy(energyA, energyB, synergy_factor)
                ct_i.energy[synergy_indices[0]] = new_energyA
                ct_j.energy[synergy_indices[1]] = new_energyB

        # Apply friction and global temperature effects vectorized
        friction_mask = np.full(n_i, self.config.friction)
        ct_i.vx *= friction_mask
        ct_i.vy *= friction_mask
        
        thermal_noise = np.random.uniform(-0.5, 0.5, n_i) * self.config.global_temperature
        ct_i.vx += thermal_noise
        ct_i.vy += thermal_noise

        # Update positions vectorized
        ct_i.x += ct_i.vx
        ct_i.y += ct_i.vy

        # Handle boundary conditions
        self.handle_boundary_reflections(ct_i)

        # Age components and update states vectorized
        ct_i.age_components()
        ct_i.update_states()
        ct_i.update_alive()

    def handle_boundary_reflections(self, ct: Optional[CellularTypeData] = None) -> None:
        """
        Handle boundary reflections for cellular components using vectorized operations.
        """
        cellular_types = [ct] if ct else self.type_manager.cellular_types

        for ct in cellular_types:
            if ct.x.size == 0:
                continue

            # Create boolean masks for boundary violations
            left_mask = ct.x < self.screen_bounds[0]
            right_mask = ct.x > self.screen_bounds[1]
            top_mask = ct.y < self.screen_bounds[2]
            bottom_mask = ct.y > self.screen_bounds[3]

            # Reflect velocities where needed
            ct.vx[left_mask | right_mask] *= -1
            ct.vy[top_mask | bottom_mask] *= -1

            # Clamp positions to bounds
            np.clip(ct.x, self.screen_bounds[0], self.screen_bounds[1], out=ct.x)
            np.clip(ct.y, self.screen_bounds[2], self.screen_bounds[3], out=ct.y)

    def cull_oldest_particles(self) -> None:
        """
        Advanced adaptive particle management system that aggressively optimizes performance
        through multi-factor analysis and predictive culling.
        """
        # Initialize performance tracking metrics if not present
        if not hasattr(self, '_performance_metrics'):
            self._performance_metrics = {
                'fps_history': collections.deque(maxlen=60),
                'particle_counts': collections.deque(maxlen=60), 
                'cull_history': collections.deque(maxlen=10),
                'last_cull_time': time.time(),
                'performance_score': 1.0,
                'stress_threshold': 0.7,
                'min_fps': 45,
                'target_fps': 90,
                'emergency_fps': 30,
                'last_emergency': 0
            }

        metrics = self._performance_metrics
        current_time = time.time()
        current_fps = self.clock.get_fps()
        
        # Track metrics
        metrics['fps_history'].append(current_fps)
        total_particles = sum(ct.x.size for ct in self.type_manager.cellular_types)
        metrics['particle_counts'].append(total_particles)
        
        # Calculate performance indicators
        avg_fps = np.mean(metrics['fps_history'])
        fps_trend = np.gradient(list(metrics['fps_history']))[-10:].mean()
        particle_trend = np.gradient(list(metrics['particle_counts']))[-10:].mean()
        
        # System stress calculation (0-1 scale)
        fps_stress = max(0, (metrics['target_fps'] - avg_fps) / metrics['target_fps'])
        particle_stress = sigmoid(total_particles / 10000)  # Normalize particle count
        system_stress = (fps_stress * 0.7 + particle_stress * 0.3)
        
        # Emergency protocol for severe performance degradation
        if (current_fps < metrics['emergency_fps'] and 
            current_time - metrics['last_emergency'] > 5.0):
            emergency_cull_factor = 0.5  # Remove 50% of particles
            metrics['last_emergency'] = current_time
            metrics['performance_score'] *= 2.0
            
            for ct in self.type_manager.cellular_types:
                if ct.x.size < 50:  # Preserve minimal population
                    continue
                    
                keep_count = max(50, int(ct.x.size * (1 - emergency_cull_factor)))
                self._emergency_cull(ct, keep_count)
            return

        # Adaptive performance scoring
        if avg_fps < metrics['min_fps']:
            metrics['performance_score'] *= 1.5
        elif avg_fps < metrics['target_fps']:
            metrics['performance_score'] *= 1.2
        elif avg_fps > metrics['target_fps']:
            metrics['performance_score'] = max(0.2, metrics['performance_score'] * 0.9)
            
        # Factor in trends
        if fps_trend < 0:  # FPS declining
            metrics['performance_score'] *= 1.2
        if particle_trend > 0:  # Particle count increasing
            metrics['performance_score'] *= 1.1
            
        metrics['performance_score'] = np.clip(metrics['performance_score'], 0.2, 10.0)
        
        # Process each cellular type
        for ct in self.type_manager.cellular_types:
            if ct.x.size < 100:
                continue
                
            # Build spatial index
            positions = np.column_stack((ct.x, ct.y))
            tree = cKDTree(positions)
            
            # Calculate advanced fitness metrics
            fitness_scores = np.zeros(ct.x.size)
            
            # Density analysis
            density_scores = tree.query_ball_point(positions, r=50, return_length=True)
            density_penalty = density_scores / (np.max(density_scores) + 1e-6)
            
            # Energy efficiency score
            energy_score = ct.energy * ct.energy_efficiency * (1 - (ct.age / ct.max_age))
            
            # Interaction value
            interaction_score = (ct.interaction_strength * ct.synergy_affinity * 
                               ct.colony_factor * ct.reproduction_rate)
            
            # Combine scores with weighted importance
            fitness_scores = (
                energy_score * 0.4 +
                interaction_score * 0.3 +
                (1 - density_penalty) * 0.3
            )
            
            # Normalize scores
            fitness_scores = (fitness_scores - np.min(fitness_scores)) / (np.max(fitness_scores) - np.min(fitness_scores) + 1e-10)
            
            # Calculate adaptive cull rate
            base_cull_rate = 0.1 * metrics['performance_score'] * system_stress
            cull_rate = np.clip(base_cull_rate, 0.05, 0.4)  # 5% to 40% removal
            removal_count = int(ct.x.size * cull_rate)
            
            # Keep highest fitness particles
            keep_indices = np.argsort(fitness_scores)[removal_count:]
            keep_mask = np.zeros(ct.x.size, dtype=bool)
            keep_mask[keep_indices] = True
            
            # Apply culling to all arrays
            arrays_to_filter = [
                'x', 'y', 'vx', 'vy', 'energy', 'alive', 'age', 'energy_efficiency',
                'speed_factor', 'interaction_strength', 'perception_range',
                'reproduction_rate', 'synergy_affinity', 'colony_factor',
                'drift_sensitivity', 'species_id', 'parent_id'
            ]
            
            for attr in arrays_to_filter:
                if hasattr(ct, attr):
                    setattr(ct, attr, getattr(ct, attr)[keep_mask])
                    
            if ct.mass_based and ct.mass is not None:
                ct.mass = ct.mass[keep_mask]
                
        metrics['last_cull_time'] = current_time
        
    def _emergency_cull(self, ct: CellularTypeData, keep_count: int) -> None:
        """Emergency culling for severe performance degradation"""
        indices = np.argsort(ct.energy * ct.energy_efficiency)[-keep_count:]
        mask = np.zeros(ct.x.size, dtype=bool)
        mask[indices] = True
        
        for attr in ['x', 'y', 'vx', 'vy', 'energy', 'alive', 'age', 'energy_efficiency',
                    'speed_factor', 'interaction_strength', 'perception_range',
                    'reproduction_rate', 'synergy_affinity', 'colony_factor', 
                    'drift_sensitivity', 'species_id', 'parent_id']:
            if hasattr(ct, attr):
                setattr(ct, attr, getattr(ct, attr)[mask])
        
        if ct.mass_based and ct.mass is not None:
            ct.mass = ct.mass[mask]

def sigmoid(x: float) -> float:
    """Sigmoid function for smooth normalization"""
    return 1 / (1 + np.exp(-x))

    def apply_clustering(self, ct: CellularTypeData) -> None:
        """
        Apply clustering forces within a single cellular type using KD-Tree for efficiency.
        Utilizes colony_factor to influence colony formation and cohesion.

        Parameters:
        -----------
        ct : CellularTypeData
            The cellular type data of the particle.
        """
        n = ct.x.size
        if n < 2:
            return

        # Build KD-Tree once for position data
        positions = np.column_stack((ct.x, ct.y))
        tree = cKDTree(positions)
        
        # Query all neighbors at once
        indices = tree.query_ball_tree(tree, self.config.cluster_radius)
        
        # Pre-allocate velocity change arrays
        dvx = np.zeros(n)
        dvy = np.zeros(n)
        
        # Vectorized calculations for all components
        for idx, neighbor_indices in enumerate(indices):
            neighbor_indices = [i for i in neighbor_indices if i != idx and ct.alive[i]]
            if not neighbor_indices:
                continue
                
            neighbor_positions = positions[neighbor_indices]
            neighbor_velocities = np.column_stack((ct.vx[neighbor_indices], ct.vy[neighbor_indices]))
            
            # Alignment
            avg_velocity = np.mean(neighbor_velocities, axis=0)
            alignment = (avg_velocity - np.array([ct.vx[idx], ct.vy[idx]])) * self.config.alignment_strength
            
            # Cohesion influenced by colony_factor
            avg_position = np.mean(neighbor_positions, axis=0)
            cohesion = (avg_position - positions[idx]) * self.config.cohesion_strength * ct.colony_factor[idx]
            
            # Separation
            separation = (positions[idx] - np.mean(neighbor_positions, axis=0)) * self.config.separation_strength
            
            # Combine forces
            total_force = alignment + cohesion + separation
            dvx[idx] = total_force[0]
            dvy[idx] = total_force[1]
        
        # Apply accumulated velocity changes
        ct.vx += dvx
        ct.vy += dvy

###############################################################
# Renderer Class
###############################################################

class Renderer:
    """
    Handles rendering of cellular components on the Pygame window.
    """
    
    def __init__(self, surface: pygame.Surface, config: SimulationConfig):
        """
        Initialize the Renderer with a Pygame surface and configuration.

        Parameters:
        -----------
        surface : pygame.Surface
            The Pygame surface where cellular components will be drawn.
        config : SimulationConfig
            Configuration parameters for the simulation.
        """
        self.surface = surface  # Pygame surface for rendering
        self.config = config    # Simulation configuration

        # Precompute a surface for all particles to optimize rendering
        self.particle_surface = pygame.Surface(self.surface.get_size(), flags=pygame.SRCALPHA)
        self.particle_surface = self.particle_surface.convert_alpha()

        # Initialize font for rendering statistics
        pygame.font.init()  # Initialize Pygame font module
        self.font = pygame.font.SysFont('Arial', 20)  # Use Arial font, size 20

    def draw_component(self, x: float, y: float, color: Tuple[int, int, int], energy: float, speed_factor: float) -> None:
        """
        Draw a single cellular component as a circle on the surface. Energy is mapped to brightness.

        Parameters:
        -----------
        x : float
            X-coordinate of the component.
        y : float
            Y-coordinate of the component.
        color : Tuple[int, int, int]
            RGB color tuple of the component.
        energy : float
            Current energy level of the component, influencing brightness.
        speed_factor : float
            Speed factor gene trait influencing the brightness scaling.
        """
        # Clamp energy between 0 and 100 for brightness scaling
        health = min(100.0, max(0.0, energy))
        # Normalize energy to [0,1] for intensity factor
        intensity_factor = max(0.0, min(1.0, health / 100.0))

        # Adjust color brightness based on energy and speed_factor
        # Higher energy and speed_factor -> brighter color; lower values -> dimmer color with a base minimum brightness
        c = (
            min(255, int(color[0] * intensity_factor * speed_factor + (1 - intensity_factor) * 100)),
            min(255, int(color[1] * intensity_factor * speed_factor + (1 - intensity_factor) * 100)),
            min(255, int(color[2] * intensity_factor * speed_factor + (1 - intensity_factor) * 100))
        )

        # Draw the component as a filled circle with the calculated color and size
        pygame.draw.circle(self.particle_surface, c, (int(x), int(y)), int(self.config.particle_size))

    def draw_cellular_type(self, ct: CellularTypeData) -> None:
        """
        Draw all alive cellular components of a cellular type on the surface.

        Parameters:
        -----------
        ct : CellularTypeData
            The cellular type data containing components to draw.
        """
        # Utilize NumPy's vectorization to efficiently iterate over alive components
        alive_indices = np.where(ct.alive)[0]  # Get indices of alive components
        # Iterate over alive components and draw them
        for idx in alive_indices:
            self.draw_component(ct.x[idx], ct.y[idx], ct.color, ct.energy[idx], ct.speed_factor[idx])

    def render(self, stats: Dict[str, Any]) -> None:
        """
        Blit the particle surface onto the main surface and reset the particle surface for the next frame.
        Additionally, render simulation statistics on the screen.

        Parameters:
        -----------
        stats : Dict[str, Any]
            Dictionary containing simulation statistics to display.
        """
        self.surface.blit(self.particle_surface, (0, 0))  # Blit all particles at once
        self.particle_surface.fill((0, 0, 0, 0))  # Clear particle surface for next frame

        # Render statistics on the top-left corner
        stats_text = f"FPS: {stats.get('fps', 0):.2f} | Total Species: {stats.get('total_species', 0)} | Total Particles: {stats.get('total_particles', 0)}"
        text_surface = self.font.render(stats_text, True, (255, 255, 255))  # Render text in white
        self.surface.blit(text_surface, (10, 10))  # Position text at (10,10)

###############################################################
# Cellular Automata (Main Simulation)
###############################################################

class CellularAutomata:
    """
    The main simulation class. Initializes and runs the simulation loop.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the CellularAutomata with the given configuration.

        Parameters:
        -----------
        config : SimulationConfig
            Configuration parameters for the simulation.
        """
        self.config = config  # Store simulation configuration
        pygame.init()  # Initialize all imported Pygame modules

        # Retrieve display information to set fullscreen window
        display_info = pygame.display.Info()
        screen_width, screen_height = display_info.current_w, display_info.current_h  # Current display dimensions

        # Set up a fullscreen window with the calculated dimensions
        self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)  # Create optimized fullscreen window
        pygame.display.set_caption("Emergent Cellular Automata Simulation")  # Set window title

        self.clock = pygame.time.Clock()  # Clock to manage frame rate
        self.frame_count = 0  # Initialize frame counter
        self.run_flag = True  # Flag to control main loop

        # Calculate minimum distance from edges (5% of the larger screen dimension)
        self.edge_buffer = 0.05 * max(screen_width, screen_height)  # Minimum distance from edges in pixels

        # Setup cellular type colors and identify mass-based types
        self.colors = generate_vibrant_colors(self.config.n_cell_types)  # Generate colors for cellular types
        n_mass_types = int(self.config.mass_based_fraction * self.config.n_cell_types)  # Number of mass-based types
        mass_based_type_indices = list(range(n_mass_types))  # Indices of mass-based types

        # Initialize CellularTypeManager with configuration, colors, and mass-based type indices
        self.type_manager = CellularTypeManager(self.config, self.colors, mass_based_type_indices)

        # Pre-calculate mass values for efficiency
        mass_values = np.random.uniform(self.config.mass_range[0], self.config.mass_range[1], n_mass_types)

        # Create cellular type data for each type and add to CellularTypeManager
        for i in range(self.config.n_cell_types):
            ct = CellularTypeData(
                type_id=i,
                color=self.colors[i],
                n_particles=self.config.particles_per_type,
                window_width=screen_width,
                window_height=screen_height,
                initial_energy=self.config.initial_energy,
                max_age=self.config.max_age,
                mass=mass_values[i] if i < n_mass_types else None,
                base_velocity_scale=self.config.base_velocity_scale
            )
            self.type_manager.add_cellular_type_data(ct)  # Add type to manager

        # Initialize InteractionRules with configuration and mass-based type indices
        self.rules_manager = InteractionRules(self.config, mass_based_type_indices)
        
        # Initialize Renderer with Pygame surface and configuration
        self.renderer = Renderer(self.screen, self.config)

        # Initialize statistics tracking using NumPy array for better performance
        self.species_count = defaultdict(int)  # Tracks number of species
        self.update_species_count()  # Initialize species count

        # Pre-allocate arrays for boundary calculations
        self.screen_bounds = np.array([
            self.edge_buffer,
            screen_width - self.edge_buffer,
            self.edge_buffer,
            screen_height - self.edge_buffer
        ])

    def update_species_count(self) -> None:
        """
        Update the species count based on current cellular types.
        """
        self.species_count.clear()  # Reset species count
        for ct in self.type_manager.cellular_types:
            unique, counts = np.unique(ct.species_id, return_counts=True)
            for species, count in zip(unique, counts):
                self.species_count[species] += count

    def main_loop(self) -> None:
        """
        Run the main simulation loop until exit conditions are met.
        """
        while self.run_flag:
            self.frame_count += 1  # Increment frame counter
            if self.config.max_frames > 0 and self.frame_count > self.config.max_frames:
                self.run_flag = False  # Exit if max_frames is reached

            # Handle Pygame events (e.g., window close, key presses)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.run_flag = False  # Exit on quit event or ESC key
                    break

            # Evolve interaction parameters periodically
            self.rules_manager.evolve_parameters(self.frame_count)

            # Clear the main screen by filling it with dark gray to prevent piling up
            self.screen.fill((69, 69, 69))  # Fill the main screen with dark gray

            # Apply all inter-type interactions and updates
            self.apply_all_interactions()

            # Apply clustering within each cellular type to simulate flocking behavior
            for ct in self.type_manager.cellular_types:
                self.apply_clustering(ct)

            # Handle reproduction of cellular components across all types
            self.type_manager.reproduce()

            # Remove dead cellular components from all types
            self.type_manager.remove_dead_in_all_types()

            # Update species count after reproduction and death
            self.update_species_count()

            # Draw all cellular types' components on the particle surface
            for ct in self.type_manager.cellular_types:
                self.renderer.draw_cellular_type(ct)

            # Handle boundary reflections after updating positions
            self.handle_boundary_reflections()

            # Compile statistics for rendering
            stats = {
                "fps": self.clock.get_fps(),
                "total_species": len(self.species_count),
                "total_particles": sum(self.species_count.values())
            }

            # Render all particles and statistics onto the main screen
            self.renderer.render(stats)

            # Update the full display surface to the screen
            pygame.display.flip()

            # Cap the frame rate and retrieve the current FPS
            current_fps = self.clock.tick(120)  # Attempt to cap at 120 FPS
            
            # Call the FPS display function to update the screen
            self.display_fps(self.screen, current_fps)  # Pass the screen surface and current FPS

            # Adaptive Particle Management - only check every 10th frame
            if self.frame_count % 10 == 0:
                if any(ct.x.size > 50 for ct in self.type_manager.cellular_types) or current_fps <= 60:
                    self.cull_oldest_particles()

        pygame.quit()  # Quit Pygame when the loop ends
                
    # Real-time FPS Display
    def display_fps(self, surface: pygame.Surface, fps: float) -> None:
        """
        Displays the current FPS at the top-left corner of the screen.
    
        Args:
            surface (pygame.Surface): The Pygame surface to render the FPS on.
            fps (float): The current frames per second.
        """
        font = pygame.font.Font(None, 36)  # Use a default font with size 36
        fps_text = font.render(f"FPS: {fps:.2f}", True, (255, 255, 255))  # Render the FPS text in white
        surface.blit(fps_text, (10, 10))  # Draw the text at (10, 10) on the screen

    def apply_all_interactions(self) -> None:
        """
        Apply inter-type interactions: forces, give-take, and synergy.
        """
        # Iterate over all interaction rules between cellular type pairs
        for (i, j, params) in self.rules_manager.rules:
            self.apply_interaction_between_types(i, j, params)  # Apply interactions between types i and j

    def apply_interaction_between_types(self, i: int, j: int, params: Dict[str, Any]) -> None:
        """
        Apply interaction rules between cellular type i and cellular type j.
        This includes forces, give-take, and synergy.

        Parameters:
        -----------
        i : int
            Index of the first cellular type.
        j : int
            Index of the second cellular type.
        params : Dict[str, Any]
            Interaction parameters between cellular type i and j.
        """
        ct_i = self.type_manager.get_cellular_type_by_id(i)  # Get cellular type i
        ct_j = self.type_manager.get_cellular_type_by_id(j)  # Get cellular type j

        # Extract synergy factor and give-take relationship from interaction rules
        synergy_factor = self.rules_manager.synergy_matrix[i, j]  # Synergy factor between types i and j
        is_giver = self.rules_manager.give_take_matrix[i, j]  # Give-take relationship flag

        n_i = ct_i.x.size  # Number of components in type i
        n_j = ct_j.x.size  # Number of components in type j

        if n_i == 0 or n_j == 0:
            return  # No interaction if one type has no components

        # Prepare mass parameters if gravity is used
        if params.get("use_gravity", False):
            if (ct_i.mass_based and ct_i.mass is not None and
                ct_j.mass_based and ct_j.mass is not None):
                params["m_a"] = ct_i.mass  # Mass of type i components
                params["m_b"] = ct_j.mass  # Mass of type j components
            else:
                params["use_gravity"] = False

        # Calculate pairwise differences using broadcasting
        dx = ct_i.x[:, np.newaxis] - ct_j.x  # Shape: (n_i, n_j)
        dy = ct_i.y[:, np.newaxis] - ct_j.y  # Shape: (n_i, n_j)
        dist_sq = dx * dx + dy * dy  # Squared distances, shape: (n_i, n_j)

        # Determine which pairs are within interaction range and not overlapping
        within_range = (dist_sq > 0.0) & (dist_sq <= params["max_dist"] ** 2)  # Boolean mask

        # Extract indices of interacting pairs
        indices = np.where(within_range)
        if len(indices[0]) == 0:
            return  # No interactions within range

        # Calculate distances for interacting pairs
        dist = np.sqrt(dist_sq[indices])

        # Initialize force arrays
        fx = np.zeros_like(dist)
        fy = np.zeros_like(dist)

        # Potential-based forces
        if params.get("use_potential", True):
            pot_strength = params.get("potential_strength", 1.0)
            F_pot = pot_strength / dist
            fx += F_pot * dx[indices]
            fy += F_pot * dy[indices]

        # Gravity-based forces
        if params.get("use_gravity", False):
            gravity_factor = params.get("gravity_factor", 1.0)
            F_grav = gravity_factor * (params["m_a"][indices[0]] * params["m_b"][indices[1]]) / dist_sq[indices]
            fx += F_grav * dx[indices]
            fy += F_grav * dy[indices]

        # Accumulate forces using NumPy's add.at for atomic operations
        np.add.at(ct_i.vx, indices[0], fx)
        np.add.at(ct_i.vy, indices[0], fy)

        # Handle give-take interactions
        if is_giver:
            give_take_within = dist_sq[indices] <= self.config.predation_range ** 2
            give_take_indices = (indices[0][give_take_within], indices[1][give_take_within])
            if give_take_indices[0].size > 0:
                giver_energy = ct_i.energy[give_take_indices[0]]
                receiver_energy = ct_j.energy[give_take_indices[1]]
                giver_mass = ct_i.mass[give_take_indices[0]] if ct_i.mass_based else None
                receiver_mass = ct_j.mass[give_take_indices[1]] if ct_j.mass_based else None

                updated = give_take_interaction(
                    giver_energy,
                    receiver_energy,
                    giver_mass,
                    receiver_mass,
                    self.config
                )
                ct_i.energy[give_take_indices[0]] = updated[0]
                ct_j.energy[give_take_indices[1]] = updated[1]

                if ct_i.mass_based and ct_i.mass is not None and updated[2] is not None:
                    ct_i.mass[give_take_indices[0]] = updated[2]
                if ct_j.mass_based and ct_j.mass is not None and updated[3] is not None:
                    ct_j.mass[give_take_indices[1]] = updated[3]

        # Handle synergy interactions
        if synergy_factor > 0.0 and self.config.synergy_range > 0.0:
            synergy_within = dist_sq[indices] <= self.config.synergy_range ** 2
            synergy_indices = (indices[0][synergy_within], indices[1][synergy_within])
            if synergy_indices[0].size > 0:
                energyA = ct_i.energy[synergy_indices[0]]
                energyB = ct_j.energy[synergy_indices[1]]
                new_energyA, new_energyB = apply_synergy(energyA, energyB, synergy_factor)
                ct_i.energy[synergy_indices[0]] = new_energyA
                ct_j.energy[synergy_indices[1]] = new_energyB

        # Apply friction and global temperature effects vectorized
        friction_mask = np.full(n_i, self.config.friction)
        ct_i.vx *= friction_mask
        ct_i.vy *= friction_mask
        
        thermal_noise = np.random.uniform(-0.5, 0.5, n_i) * self.config.global_temperature
        ct_i.vx += thermal_noise
        ct_i.vy += thermal_noise

        # Update positions vectorized
        ct_i.x += ct_i.vx
        ct_i.y += ct_i.vy

        # Handle boundary conditions
        self.handle_boundary_reflections(ct_i)

        # Age components and update states vectorized
        ct_i.age_components()
        ct_i.update_states()
        ct_i.update_alive()

    def handle_boundary_reflections(self, ct: Optional[CellularTypeData] = None) -> None:
        """
        Handle boundary reflections for cellular components using vectorized operations.
        """
        cellular_types = [ct] if ct else self.type_manager.cellular_types

        for ct in cellular_types:
            if ct.x.size == 0:
                continue

            # Create boolean masks for boundary violations
            left_mask = ct.x < self.screen_bounds[0]
            right_mask = ct.x > self.screen_bounds[1]
            top_mask = ct.y < self.screen_bounds[2]
            bottom_mask = ct.y > self.screen_bounds[3]

            # Reflect velocities where needed
            ct.vx[left_mask | right_mask] *= -1
            ct.vy[top_mask | bottom_mask] *= -1

            # Clamp positions to bounds
            np.clip(ct.x, self.screen_bounds[0], self.screen_bounds[1], out=ct.x)
            np.clip(ct.y, self.screen_bounds[2], self.screen_bounds[3], out=ct.y)

    def cull_oldest_particles(self) -> None:
        """
        Efficiently cull the oldest particles across all cellular types.
        """
        for ct in self.type_manager.cellular_types:
            if ct.x.size < 500:
                continue

            # Find indices of particles to keep (all except oldest)
            keep_mask = np.ones(ct.x.size, dtype=bool)
            keep_mask[np.argmax(ct.age)] = False
            
            # Apply mask to all arrays at once
            ct.x = ct.x[keep_mask]
            ct.y = ct.y[keep_mask]
            ct.vx = ct.vx[keep_mask]
            ct.vy = ct.vy[keep_mask]
            ct.energy = ct.energy[keep_mask]
            ct.alive = ct.alive[keep_mask]
            ct.age = ct.age[keep_mask]
            ct.energy_efficiency = ct.energy_efficiency[keep_mask]
            ct.speed_factor = ct.speed_factor[keep_mask]
            ct.interaction_strength = ct.interaction_strength[keep_mask]
            if ct.mass_based and ct.mass is not None:
                ct.mass = ct.mass[keep_mask]

    def add_global_energy(self) -> None:
        """
        Add energy to all particles efficiently using vectorized operations.
        """
        for ct in self.type_manager.cellular_types:
            ct.energy = np.clip(ct.energy * 1.1, 0.0, 200.0)

    def apply_clustering(self, ct: CellularTypeData) -> None:
        """
        Apply clustering forces within a single cellular type using KD-Tree for efficiency.
        """
        n = ct.x.size
        if n < 2:
            return

        # Build KD-Tree once for position data
        positions = np.column_stack((ct.x, ct.y))
        tree = cKDTree(positions)
        
        # Query all neighbors at once
        indices = tree.query_ball_tree(tree, self.config.cluster_radius)
        
        # Pre-allocate velocity change arrays
        dvx = np.zeros(n)
        dvy = np.zeros(n)
        
        # Vectorized calculations for all components
        for idx, neighbor_indices in enumerate(indices):
            neighbor_indices = [i for i in neighbor_indices if i != idx and ct.alive[i]]
            if not neighbor_indices:
                continue
                
            neighbor_positions = positions[neighbor_indices]
            neighbor_velocities = np.column_stack((ct.vx[neighbor_indices], ct.vy[neighbor_indices]))
            
            # Alignment
            avg_velocity = np.mean(neighbor_velocities, axis=0)
            alignment = (avg_velocity - np.array([ct.vx[idx], ct.vy[idx]])) * self.config.alignment_strength
            
            # Cohesion
            center = np.mean(neighbor_positions, axis=0)
            cohesion = (center - positions[idx]) * self.config.cohesion_strength
            
            # Separation
            separation = (positions[idx] - np.mean(neighbor_positions, axis=0)) * self.config.separation_strength
            
            # Combine forces
            total_force = alignment + cohesion + separation
            dvx[idx] = total_force[0]
            dvy[idx] = total_force[1]
        
        # Apply accumulated velocity changes
        ct.vx += dvx
        ct.vy += dvy

###############################################################
# Entry Point
###############################################################

def main():
    """
    Main configuration and run function.
    """
    config = SimulationConfig()
    cellular_automata = CellularAutomata(config)
    cellular_automata.main_loop()

if __name__ == "__main__":
    main()
