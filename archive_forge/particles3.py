"""
Ultra-Advanced Emergent Cellular Automata and Multicellular Simulation - Fully Configurable Version
----------------------------------------------------------------------------------------------------
This simulation offers unparalleled configurability, allowing comprehensive adjustments to all aspects
of the simulation via the SimulationConfig class. Features can be toggled on or off, and every parameter
is documented with detailed descriptions, units, and constraints.

**Key Enhancements:**
1. Fully Parameterized Configuration Class
2. Comprehensive Feature Toggles
3. Optimized Performance and Scalability
4. Detailed Technical Documentation

**Dependencies:**
------------
- Python 3.x
- NumPy
- Pygame
- SciPy
- scikit-learn

**Usage:**
------
Ensure required libraries are installed:
    pip install numpy pygame scipy scikit-learn

Run the simulation:
    python fully_configurable_cellular_simulation.py

Terminate the simulation by closing the Pygame window or pressing the ESC key.

**Note:**
-----
This simulation efficiently handles tens of thousands of cellular components, showcasing emergent behaviors
through dynamic gene interactions, colony formation, resource management, and realistic environmental cycles.
All aspects of the simulation are now fully configurable, ensuring adaptability and extensibility for diverse scenarios.
"""

# Import standard and third-party libraries
import math
import random
import pygame
import pygame.gfxdraw
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
import logging

# Configure logging with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


###############################################################
# Configuration Class
###############################################################

@dataclass
class SimulationConfig:
    """
    Centralized configuration for the Cellular Automata simulation, allowing full parameterization and toggling of features.
    
    Each parameter is documented with its impact, units, constraints, typical values, and default settings.
    """

    # ======================
    # General Parameters
    # ======================

    edge_buffer: float = field(
        default=20.0,
        metadata={
            "description": "Buffer distance from window edges.",
            "impact": "Prevents components from being rendered on the edges of the window.",
            "units": "Pixels",
            "min": 0.0,
            "max": 50.0,
            "typical": 20.0,
            "default": 20.0
        }
    )
    n_cell_types: int = field(
        default=5,
        metadata={
            "description": "Number of distinct cellular types in the simulation.",
            "impact": "Determines diversity and interactions among different cellular types.",
            "units": "Count (integer)", 
            "min": 1,
            "max": 1000,
            "typical": 20,
            "default": 200
        }
    )
    particles_per_type: int = field(
        default=5000,
        metadata={
            "description": "Initial number of particles per cellular type.",
            "impact": "Sets the initial population size for each cellular type.",
            "units": "Count (integer)",
            "min": 1,
            "max": 1000,
            "typical": 200,
            "default": 5
        }
    )
    mass_range: Tuple[float, float] = field(
        default=(0.5, 5.0),
        metadata={
            "description": "Range of masses for mass-based cellular types.",
            "impact": "Defines the possible mass values for cellular types that consider mass.",
            "units": "Mass units (arbitrary)",
            "min": 0.1,
            "max": 10.0,
            "typical": (0.5, 5.0),
            "default": (0.5, 5.0)
        }
    )
    base_velocity_scale: float = field(
        default=1.0,
        metadata={
            "description": "Base scaling factor for initial velocities of cellular components.",
            "impact": "Influences the speed of cellular components upon initialization.",
            "units": "Scalar (dimensionless)",
            "min": 0.1,
            "max": 2.0,
            "typical": 0.8,
            "default": 0.8
        }
    )
    mass_based_fraction: float = field(
        default=1.0,
        metadata={
            "description": "Fraction of cellular types that are mass-based.",
            "impact": "Determines how many cellular types consider mass in their behavior.",
            "units": "Fraction (0.0 to 1.0)",
            "min": 0.0,
            "max": 1.0,
            "typical": 0.7,
            "default": 1.0
        }
    )
    interaction_strength_range: Tuple[float, float] = field(
        default=(0, 1.5),
        metadata={
            "description": "Range for interaction strength between cellular types.",
            "impact": "Defines the potential strength and direction of interactions (attraction/repulsion).",
            "units": "Force units (arbitrary)",
            "min": -2.0,
            "max": 2.0,
            "typical": (-1.0, 1.5),
            "default": (0, 1.5)
        }
    )
    max_frames: int = field(
        default=0,
        metadata={
            "description": "Maximum number of frames to run the simulation.",
            "impact": "Limits the duration of the simulation; 0 means run indefinitely.",
            "units": "Count (integer)",
            "min": 0,
            "max": 1000000,
            "typical": 0,
            "default": 0
        }
    )
    initial_energy: float = field(
        default=80.0,
        metadata={
            "description": "Initial energy level assigned to each cellular component.",
            "impact": "Sets the starting energy for cellular components, influencing their survivability and reproduction.",
            "units": "Energy units (arbitrary)",
            "min": 20.0,
            "max": 200.0,
            "typical": 80.0,
            "default": 80.0
        }
    )
    friction: float = field(
        default=0.2,
        metadata={
            "description": "Friction coefficient applied to cellular velocities each frame.",
            "impact": "Reduces velocities over time, simulating resistance.",
            "units": "Scalar (dimensionless)",
            "min": 0.001,
            "max": 1.0,
            "typical": 0.09,
            "default": 0.09
        }
    )
    global_temperature: float = field(
        default=0.5,
        metadata={
            "description": "Global temperature affecting random velocity fluctuations.",
            "impact": "Introduces randomness in cellular movements.",
            "units": "Scalar (dimensionless)",
            "min": 0.05,
            "max": 0.5,
            "typical": 0.15,
            "default": 0.15
        }
    )
    predation_range: float = field(
        default=35.0,
        metadata={
            "description": "Range within which predation (energy transfer) can occur.",
            "impact": "Determines the proximity required for energy transfer between components.",
            "units": "Pixels",
            "min": 15.0,
            "max": 100.0,
            "typical": 35.0,
            "default": 35.0
        }
    )
    energy_transfer_factor: float = field(
        default=0.3,
        metadata={
            "description": "Fraction of energy transferred during interactions.",
            "impact": "Controls how much energy is moved between cellular components during interactions.",
            "units": "Fraction (0.0 to 1.0)",
            "min": 0.1,
            "max": 0.5,
            "typical": 0.3,
            "default": 0.3
        }
    )
    mass_transfer: bool = field(
        default=True,
        metadata={
            "description": "Enable or disable mass transfer during interactions.",
            "impact": "Toggles whether mass is transferred alongside energy between components.",
            "units": "Boolean",
            "default": True
        }
    )
    max_age: float = field(
        default=np.inf,
        metadata={
            "description": "Maximum lifespan of cellular components in frames.",
            "impact": "Determines how long cellular components can exist before dying of old age.",
            "units": "Frames",
            "min": 1000.0,
            "max": 50000.0,
            "typical": 20000.0,
            "default": np.inf # Disable aging
        }
    )
    synergy_range: float = field(
        default=100.0,
        metadata={
            "description": "Range within which energy synergy occurs between allied components.",
            "impact": "Defines the proximity required for energy sharing among allied components.",
            "units": "Pixels",
            "min": 25.0,
            "max": 200.0,
            "typical": 100.0,
            "default": 100.0
        }
    )

    # ======================
    # Feature Toggles
    # ======================

    enable_max_age: bool = field(
        default=False,
        metadata={
            "description": "Enable or disable maximum age for cellular components.",
            "impact": "Toggles whether cellular components have a lifespan.",
            "units": "Boolean",
            "default": False
        }
    )
    enable_culling: bool = field(
        default=True,
        metadata={
            "description": "Enable or disable dynamic culling of cellular components.",
            "impact": "Toggles the adaptive removal of cellular components to maintain performance.",
            "units": "Boolean",
            "default": True
        }
    )

    # ======================
    # Reproduction
    # ======================

    reproduction_energy_threshold: float = field(
        default=120.0,
        metadata={
            "description": "Energy threshold required for cellular components to reproduce.",
            "impact": "Determines the minimum energy a component must have to be eligible for reproduction.",
            "units": "Energy units (arbitrary)",
            "min": 80.0,
            "max": 200.0,
            "typical": 120.0,
            "default": 120.0
        }
    )
    reproduction_mutation_rate: float = field(
        default=0.15,
        metadata={
            "description": "Probability of mutation occurring during reproduction.",
            "impact": "Controls the likelihood that offspring will undergo genetic mutations.",
            "units": "Fraction (0.0 to 1.0)",
            "min": 0.05,
            "max": 0.3,
            "typical": 0.15,
            "default": 0.15
        }
    )
    reproduction_offspring_energy_fraction: float = field(
        default=0.6,
        metadata={
            "description": "Fraction of parent energy allocated to offspring.",
            "impact": "Determines how much energy is passed from parent to offspring during reproduction.",
            "units": "Fraction (0.0 to 1.0)",
            "min": 0.4,
            "max": 0.8,
            "typical": 0.6,
            "default": 0.6
        }
    )

    # ======================
    # Clustering Behavior
    # ======================

    alignment_strength: float = field(
        default=0.4,
        metadata={
            "description": "Strength of velocity alignment in clustering behavior.",
            "impact": "Controls how strongly components align their velocities with neighbors.",
            "units": "Scalar (dimensionless)",
            "min": 0.01,
            "max": 2,
            "typical": 0.04,
            "default": 0.4
        }
    )
    cohesion_strength: float = field(
        default=0.03,
        metadata={
            "description": "Strength of cohesion in clustering behavior.",
            "impact": "Determines how strongly components move towards the center of their neighbors.",
            "units": "Scalar (dimensionless)",
            "min": 0.01,
            "max": 2,
            "typical": 0.03,
            "default": 0.3
        }
    )
    separation_strength: float = field(
        default=0.5,
        metadata={
            "description": "Strength of separation in clustering behavior.",
            "impact": "Controls how strongly components avoid overcrowding by separating from neighbors.",
            "units": "Scalar (dimensionless)",
            "min": 0.02,
            "max": 2,
            "typical": 0.08,
            "default": 0.8
        }
    )
    cluster_radius: float = field(
        default=20.0,
        metadata={
            "description": "Radius defining neighborhood for clustering behavior.",
            "impact": "Sets the proximity within which components consider others as neighbors for clustering.",
            "units": "Pixels",
            "min": 50.0,
            "max": 500.0,
            "typical": 150.0,
            "default": 150.0
        }
    )

    # ======================
    # Rendering
    # ======================

    particle_size: float = field(
        default=3.0,
        metadata={
            "description": "Radius of cellular component particles for rendering.",
            "impact": "Determines the visual size of cellular components on the screen.",
            "units": "Pixels",
            "min": 2.0,
            "max": 10.0,
            "typical": 4.0,
            "default": 4.0
        }
    )
    font_size: int = field(
        default=16,
        metadata={
            "description": "Font size for on-screen text elements like FPS display.",
            "impact": "Controls the readability of textual information rendered on the screen.",
            "units": "Points",
            "min": 12,
            "max": 24,
            "typical": 16,
            "default": 16
        }
    )

    # ======================
    # Energy Efficiency Traits
    # ======================

    energy_efficiency_range: Tuple[float, float] = field(
        default=(0.7, 1.5),
        metadata={
            "description": "Range of energy efficiency traits for cellular components.",
            "impact": "Influences how efficiently components utilize energy, affecting speed and interaction strength.",
            "units": "Scalar (dimensionless)",
            "min": 0.5,
            "max": 2.0,
            "typical": (0.7, 1.5),
            "default": (0.7, 1.5)
        }
    )
    energy_efficiency_mutation_rate: float = field(
        default=0.08,
        metadata={
            "description": "Mutation probability for energy efficiency traits during reproduction.",
            "impact": "Controls the likelihood that offspring will have altered energy efficiency traits.",
            "units": "Fraction (0.0 to 1.0)",
            "min": 0.02,
            "max": 0.2,
            "typical": 0.08,
            "default": 0.08
        }
    )
    energy_efficiency_mutation_range: Tuple[float, float] = field(
        default=(-0.15, 0.15),
        metadata={
            "description": "Range of mutation adjustments for energy efficiency traits.",
            "impact": "Determines how much energy efficiency traits can vary during mutation.",
            "units": "Scalar (dimensionless)",
            "min": -0.2,
            "max": 0.2,
            "typical": (-0.15, 0.15),
            "default": (-0.15, 0.15)
        }
    )

    # ======================
    # Environmental Zones
    # ======================

    zones: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "x": 0.3,
                "y": 0.3,
                "radius": 100.0,
                "effect": {"energy_modifier": 1.02}
            },
            {
                "x": 0.7,
                "y": 0.7,
                "radius": 100.0,
                "effect": {"energy_modifier": 0.98}
            }
        ],
        metadata={
            "description": "List of environmental zones with positions, radii, and effects.",
            "impact": "Defines regions in the simulation where cellular components experience modified energy levels.",
            "units": "Relative positions (0.0 to 1.0) for x and y; pixels for radius.",
            "default": [
                {
                    "x": 0.3,
                    "y": 0.3,
                    "radius": 100.0,
                    "effect": {"energy_modifier": 1.02}
                },
                {
                    "x": 0.7,
                    "y": 0.7,
                    "radius": 100.0,
                    "effect": {"energy_modifier": 0.98}
                }
            ]
        }
    )

    # ======================
    # Gene Configuration
    # ======================

    gene_traits: List[str] = field(
        default_factory=lambda: ["speed_factor", "interaction_strength"],
        metadata={
            "description": "List of gene trait identifiers for cellular components.",
            "impact": "Defines the traits that can mutate and influence component behavior.",
            "units": "List of strings",
            "default": ["speed_factor", "interaction_strength"]
        }
    )
    gene_mutation_rate: float = field(
        default=0.08,
        metadata={
            "description": "Mutation probability for gene traits during reproduction.",
            "impact": "Controls the likelihood that offspring will undergo mutations in their gene traits.",
            "units": "Fraction (0.0 to 1.0)",
            "min": 0.02,
            "max": 0.2,
            "typical": 0.08,
            "default": 0.08
        }
    )
    gene_mutation_range: Tuple[float, float] = field(
        default=(-0.15, 0.15),
        metadata={
            "description": "Range of mutation adjustments for gene traits.",
            "impact": "Determines how much gene traits can vary during mutation.",
            "units": "Scalar (dimensionless)",
            "min": -0.2,
            "max": 0.2,
            "typical": (-0.15, 0.15),
            "default": (-0.15, 0.15)
        }
    )

    # ======================
    # Adaptive Frame Management
    # ======================

    fps_low_threshold: float = field(
        default=45.0,
        metadata={
            "description": "FPS threshold below which culling begins to maintain performance.",
            "impact": "Triggers adaptive culling when FPS drops below this value to reduce particle count.",
            "units": "Frames per second",
            "min": 30.0,
            "max": 60.0,
            "typical": 45.0,
            "default": 45.0
        }
    )
    fps_high_threshold: float = field(
        default=90.0,
        metadata={
            "description": "FPS threshold above which energy boosting occurs to stimulate reproduction.",
            "impact": "Triggers energy boosts to cellular components when FPS exceeds this value, encouraging population growth.",
            "units": "Frames per second",
            "min": 60.0,
            "max": 144.0,
            "typical": 90.0,
            "default": 90.0
        }
    )
    cull_amount_per_frame: int = field(
        default=5,
        metadata={
            "description": "Number of particles to cull per frame when FPS is low.",
            "impact": "Defines the rate at which particles are removed to alleviate performance issues.",
            "units": "Count (integer)",
            "min": 1,
            "max": 20,
            "typical": 5,
            "default": 5
        }
    )
    energy_boost_factor: float = field(
        default=1.05,
        metadata={
            "description": "Factor by which to boost energy levels of all components when FPS is high.",
            "impact": "Enhances energy levels to promote reproduction and population growth.",
            "units": "Scalar (dimensionless)",
            "min": 1.01,
            "max": 1.2,
            "typical": 1.05,
            "default": 1.05
        }
    )
    min_total_particles: int = field(
        default=500,
        metadata={
            "description": "Minimum total number of particles to maintain in the simulation.",
            "impact": "Ensures a baseline population size is preserved even during culling.",
            "units": "Count (integer)",
            "min": 100,
            "max": 20000,
            "typical": 1000,
            "default": 500
        }
    )
    culling_ratio: float = field(
        default=0.05,
        metadata={
            "description": "Ratio determining how aggressively particles are culled relative to total population.",
            "impact": "Adjusts the proportion of particles to remove during culling to balance performance and population size.",
            "units": "Fraction (0.0 to 1.0)",
            "min": 0.01,
            "max": 0.1,
            "typical": 0.05,
            "default": 0.05
        }
    )

    # ======================
    # Culling Fitness Weights
    # ======================

    culling_fitness_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "energy_weight": 0.4,
            "age_weight": 0.8,
            "speed_factor_weight": 0.3,
            "interaction_strength_weight": 0.3
        },
        metadata={
            "description": "Weights determining the importance of various fitness factors during culling.",
            "impact": "Influences which particles are culled based on their energy, age, speed, and interaction strength.",
            "units": "Scalar (dimensionless)",
            "default": {
                "energy_weight": 0.4,
                "age_weight": 0.8,
                "speed_factor_weight": 0.3,
                "interaction_strength_weight": 0.3
            }
        }
    )

    # ======================
    # Initial Pause
    # ======================

    initial_pause_duration: int = field(
        default=500,
        metadata={
            "description": "Duration of the initial pause before the simulation starts running.",
            "impact": "Allows for initial setup and stabilization before active simulation begins.",
            "units": "Milliseconds",
            "min": 0,
            "max": 2000,
            "typical": 500,
            "default": 500
        }
    )

###############################################################
# Utility Functions
###############################################################

def random_xy(window_width: int, window_height: int, n: int = 1) -> np.ndarray:
    """
    Generate n random (x, y) coordinates within window boundaries.

    Parameters:
    -----------
    window_width : int
        Window width in pixels.
    window_height : int
        Window height in pixels.
    n : int, optional
        Number of coordinates to generate.

    Returns:
    --------
    np.ndarray
        Array of shape (n, 2) with random positions.
    """
    return np.random.uniform(0, [window_width, window_height], (n, 2)).astype(float)


def generate_vibrant_colors(n: int) -> List[str]:
    """
    Generate n distinct vibrant neon color names from a predefined set.

    Parameters:
    -----------
    n : int
        Number of colors to generate.

    Returns:
    --------
    List[str]
        List of color names.
    """
    base_colors = [
        "neon_red", "neon_green", "neon_blue", "neon_yellow", "neon_cyan",
        "neon_magenta", "neon_orange", "neon_lime", "neon_purple", "neon_pink",
        "neon_gold", "neon_aqua", "neon_chartreuse", "neon_crimson", "neon_navy",
        "neon_maroon", "neon_fuchsia", "neon_teal", "neon_electric_blue", "neon_bright_green",
        "neon_spring_green", "neon_sky_blue", "neon_pale_green", "neon_dark_purple"
    ]
    vibrant_colors = [color for color in base_colors if color not in {"grey", "gray", "white", "black", "brown", "cream"}]
    unique_vibrant_colors = list(dict.fromkeys(vibrant_colors))  # Remove duplicates while preserving order
    if len(unique_vibrant_colors) < n:
        unique_vibrant_colors += [f"{color}_{i}" for i in range(n - len(unique_vibrant_colors)) for color in vibrant_colors]
    return unique_vibrant_colors[:n]

###############################################################
# Cellular Component & Type Management
###############################################################

class CellularTypeData:
    """
    Manages cellular component data for a single type using NumPy arrays.

    Attributes:
    -----------
    type_id : int
        Unique identifier.
    color : str
        Color name for rendering.
    mass_based : bool
        Indicates if mass is considered.
    x, y : np.ndarray
        Positions.
    vx, vy : np.ndarray
        Velocities.
    energy : np.ndarray
        Energy levels.
    mass : Optional[np.ndarray]
        Masses if mass-based.
    alive : np.ndarray
        Alive status.
    age : np.ndarray
        Ages.
    max_age : float
        Maximum age.
    energy_efficiency : np.ndarray
        Energy efficiency trait.
    speed_factor : np.ndarray
        Speed factor gene trait.
    interaction_strength : np.ndarray
        Interaction strength gene trait.
    gene_mutation_rate : float
        Mutation probability for gene traits.
    gene_mutation_range : Tuple[float, float]
        Mutation adjustment bounds for gene traits.
    """

    __slots__ = [
        'type_id', 'color', 'mass_based',
        'x', 'y', 'vx', 'vy', 'energy', 'mass',
        'alive', 'age', 'max_age',
        'energy_efficiency', 'speed_factor', 'interaction_strength',
        'gene_mutation_rate', 'gene_mutation_range'
    ]

    def __init__(self,
                 type_id: int,
                 color: str,
                 n_particles: int,
                 window_width: int,
                 window_height: int,
                 config: 'SimulationConfig',
                 mass: Optional[float] = None,
                 base_velocity_scale: float = 1.0,
                 energy_efficiency: Optional[float] = None):
        """
        Initialize CellularTypeData with specified parameters.

        Parameters:
        -----------
        type_id : int
            Unique identifier.
        color : str
            Color name.
        n_particles : int
            Initial number of components.
        window_width : int
            Window width.
        window_height : int
            Window height.
        config : SimulationConfig
            Simulation configuration instance.
        mass : Optional[float], default=None
            Mass if mass-based.
        base_velocity_scale : float, default=1.0
            Base velocity scaling.
        energy_efficiency : Optional[float], default=None
            Initial energy efficiency.
        """
        self.type_id: int = type_id
        self.color: str = color
        self.mass_based: bool = mass is not None

        # Initialize positions
        coords = random_xy(window_width, window_height, n_particles)
        self.x: np.ndarray = coords[:, 0]
        self.y: np.ndarray = coords[:, 1]

        # Initialize energy efficiency
        if energy_efficiency is None:
            self.energy_efficiency: np.ndarray = np.random.uniform(
                config.energy_efficiency_range[0],
                config.energy_efficiency_range[1],
                n_particles
            )
        else:
            self.energy_efficiency: np.ndarray = np.full(n_particles, energy_efficiency, dtype=float)

        # Initialize velocities based on energy efficiency
        velocity_scale = base_velocity_scale / self.energy_efficiency
        self.vx: np.ndarray = np.random.uniform(-0.5, 0.5, n_particles) * velocity_scale
        self.vy: np.ndarray = np.random.uniform(-0.5, 0.5, n_particles) * velocity_scale

        # Initialize energy
        self.energy: np.ndarray = np.full(n_particles, config.initial_energy, dtype=float)

        # Initialize mass if applicable
        if self.mass_based:
            if mass is None or mass <= 0.0:
                raise ValueError("Mass must be positive for mass-based types.")
            self.mass: np.ndarray = np.full(n_particles, mass, dtype=float)
        else:
            self.mass = None

        # Initialize status
        self.alive: np.ndarray = np.ones(n_particles, dtype=bool)
        self.age: np.ndarray = np.zeros(n_particles, dtype=float)
        self.max_age: float = config.max_age

        # Initialize gene traits
        self.speed_factor: np.ndarray = np.random.uniform(0.5, 1.5, n_particles)
        self.interaction_strength: np.ndarray = np.random.uniform(0.5, 1.5, n_particles)

        # Mutation parameters
        self.gene_mutation_rate: float = config.gene_mutation_rate
        self.gene_mutation_range: Tuple[float, float] = config.gene_mutation_range

    def is_alive_mask(self) -> np.ndarray:
        """
        Compute alive mask based on energy, age, and mass.

        Returns:
        --------
        np.ndarray
            Boolean array where True denotes alive components.
        """
        mask = (self.alive & (self.energy > 0.0) & (self.age < self.max_age))
        if self.mass_based and self.mass is not None:
            mask &= (self.mass > 0.0)
        return mask

    def update_alive(self) -> None:
        """
        Update alive status based on current conditions.
        """
        self.alive = self.is_alive_mask()

    def age_components(self) -> None:
        """
        Increment age by one frame.
        """
        self.age += 1.0

    def update_states(self) -> None:
        """
        Placeholder for future state updates.
        """
        pass

    def remove_dead(self, config: 'SimulationConfig') -> None:
        """
        Remove dead components, redistributing energy to neighbors using cKDTree for efficiency.

        Parameters:
        -----------
        config : SimulationConfig
            Simulation configuration.
        """
        if not config.enable_culling:
            return

        dead_mask = (~self.alive) & (self.age >= self.max_age)
        dead_indices = np.where(dead_mask)[0]
        if dead_indices.size == 0:
            return

        alive_mask = self.alive.copy()
        alive_mask[dead_indices] = False
        alive_indices = np.where(alive_mask)[0]
        num_alive = alive_indices.size

        if num_alive > 0:
            dead_coords = np.vstack((self.x[dead_indices], self.y[dead_indices])).T
            alive_coords = np.vstack((self.x[alive_indices], self.y[alive_indices])).T

            tree = cKDTree(alive_coords)
            distances, neighbor_idxs = tree.query(dead_coords, k=3, distance_upper_bound=config.predation_range)

            # Identify valid neighbors within predation range
            valid = distances < config.predation_range

            # Transfer energy from dead to alive neighbors using vectorized operations
            num_valid = valid.sum(axis=1)
            valid_dead_mask = num_valid > 0
            valid_dead_indices = dead_indices[valid_dead_mask]
            selected_neighbor_idxs = neighbor_idxs[valid_dead_mask][valid[valid_dead_mask]]

            dead_energy = self.energy[valid_dead_indices]
            energy_per_neighbor = dead_energy / num_valid[valid_dead_mask]

            # Flatten neighbor indices and energy transfers
            alive_neighbor_flat = alive_indices[selected_neighbor_idxs.flatten()]
            energy_transfer_flat = np.repeat(energy_per_neighbor, valid[valid_dead_mask].sum(axis=1))

            # Transfer energy
            np.add.at(self.energy, alive_neighbor_flat, energy_transfer_flat)

        # Remove dead components
        mask = self.is_alive_mask()
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.vx = self.vx[mask]
        self.vy = self.vy[mask]
        self.energy = self.energy[mask]
        self.alive = self.alive[mask]
        self.age = self.age[mask]
        self.energy_efficiency = self.energy_efficiency[mask]
        self.speed_factor = self.speed_factor[mask]
        self.interaction_strength = self.interaction_strength[mask]
        if self.mass_based and self.mass is not None:
            self.mass = self.mass[mask]

    def add_component(self,
                      x: float,
                      y: float,
                      vx: float,
                      vy: float,
                      energy: float,
                      mass_val: Optional[float],
                      energy_efficiency_val: float,
                      speed_factor_val: float,
                      interaction_strength_val: float,
                      config: 'SimulationConfig') -> None:
        """
        Add a new component, typically as offspring.

        Parameters:
        -----------
        x, y : float
            Position coordinates.
        vx, vy : float
            Velocity components.
        energy : float
            Initial energy.
        mass_val : Optional[float]
            Mass if applicable.
        energy_efficiency_val : float
            Energy efficiency trait.
        speed_factor_val : float
            Speed factor trait.
        interaction_strength_val : float
            Interaction strength trait.
        config : SimulationConfig
            Simulation configuration instance.
        """
        # Append new component data
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.vx = np.append(self.vx, vx)
        self.vy = np.append(self.vy, vy)
        self.energy = np.append(self.energy, energy)
        self.alive = np.append(self.alive, True)
        self.age = np.append(self.age, 0.0)
        self.energy_efficiency = np.append(self.energy_efficiency, energy_efficiency_val)
        self.speed_factor = np.append(self.speed_factor, speed_factor_val)
        self.interaction_strength = np.append(self.interaction_strength, interaction_strength_val)
        if self.mass_based and self.mass is not None:
            mass_val = mass_val if (mass_val is not None and mass_val > 0.0) else 0.1
            self.mass = np.append(self.mass, mass_val)

###############################################################
# Cellular Type Manager
###############################################################

class CellularTypeManager:
    """
    Manages all cellular types, handling reproduction and interactions.

    Attributes:
    -----------
    config : SimulationConfig
        Simulation configuration.
    cellular_types : List[CellularTypeData]
        List of cellular types.
    mass_based_type_indices : List[int]
        Indices of mass-based types.
    colors : List[str]
        Color identifiers for types.
    """
    def __init__(self, config: SimulationConfig, colors: List[str], mass_based_type_indices: List[int]):
        """
        Initialize CellularTypeManager with configuration, colors, and mass-based type indices.

        Parameters:
        -----------
        config : SimulationConfig
            Simulation configuration instance.
        colors : List[str]
            Color names for types.
        mass_based_type_indices : List[int]
            Indices indicating mass-based types.
        """
        self.config = config
        self.cellular_types: List[CellularTypeData] = []
        self.mass_based_type_indices = mass_based_type_indices
        self.colors = colors

    def add_cellular_type_data(self, data: CellularTypeData) -> None:
        """
        Add a CellularTypeData instance to the manager.

        Parameters:
        -----------
        data : CellularTypeData
            Cellular type data to add.
        """
        self.cellular_types.append(data)

    def get_cellular_type_by_id(self, i: int) -> CellularTypeData:
        """
        Retrieve a CellularTypeData instance by type ID.

        Parameters:
        -----------
        i : int
            Type ID.

        Returns:
        --------
        CellularTypeData
            Corresponding cellular type data.
        """
        return self.cellular_types[i]

    def remove_dead_in_all_types(self) -> None:
        """
        Remove dead components from all cellular types.
        """
        for ct in self.cellular_types:
            ct.remove_dead(self.config)

    def reproduce(self) -> None:
        """
        Handle reproduction across all cellular types based on eligibility and mutation rates.
        """
        for ct in self.cellular_types:
            if not self.config.enable_culling:
                continue
            eligible_mask = (ct.alive & (ct.energy > self.config.reproduction_energy_threshold))
            num_offspring = np.sum(eligible_mask)
            if num_offspring == 0:
                continue

            # Halve the energy of parents
            ct.energy[eligible_mask] *= 0.5

            # Calculate offspring energy
            offspring_energy = ct.energy[eligible_mask] * self.config.reproduction_offspring_energy_fraction

            # Determine mutations
            mutation_mask = np.random.random(num_offspring) < self.config.reproduction_mutation_rate
            offspring_energy[mutation_mask] *= np.random.uniform(0.9, 1.1, size=mutation_mask.sum())

            # Energy efficiency mutations
            offspring_efficiency = ct.energy_efficiency[eligible_mask].copy()
            offspring_efficiency[mutation_mask] += np.random.uniform(
                self.config.energy_efficiency_mutation_range[0],
                self.config.energy_efficiency_mutation_range[1],
                size=mutation_mask.sum()
            )
            offspring_efficiency = np.clip(
                offspring_efficiency,
                self.config.energy_efficiency_range[0],
                self.config.energy_efficiency_range[1]
            )

            # Speed factor and interaction strength mutations
            offspring_speed_factor = ct.speed_factor[eligible_mask].copy()
            offspring_interaction_strength = ct.interaction_strength[eligible_mask].copy()

            offspring_speed_factor[mutation_mask] += np.random.uniform(
                self.config.gene_mutation_range[0],
                self.config.gene_mutation_range[1],
                size=mutation_mask.sum()
            )
            offspring_speed_factor = np.clip(
                offspring_speed_factor,
                0.1, 3.0
            )

            offspring_interaction_strength[mutation_mask] += np.random.uniform(
                self.config.gene_mutation_range[0],
                self.config.gene_mutation_range[1],
                size=mutation_mask.sum()
            )
            offspring_interaction_strength = np.clip(
                offspring_interaction_strength,
                0.1, 3.0
            )

            # Mass mutations if applicable
            if ct.mass_based and ct.mass is not None:
                offspring_mass = ct.mass[eligible_mask].copy()
                offspring_mass[mutation_mask] *= np.random.uniform(0.95, 1.05, size=mutation_mask.sum())
                offspring_mass = np.maximum(offspring_mass, 0.1)
            else:
                offspring_mass = None

            # Offspring positions (same as parents)
            offspring_x = ct.x[eligible_mask].copy()
            offspring_y = ct.y[eligible_mask].copy()

            # Offspring velocities
            offspring_velocity_scale = self.config.base_velocity_scale / offspring_efficiency * offspring_speed_factor
            offspring_vx = np.random.uniform(-0.5, 0.5, num_offspring) * offspring_velocity_scale
            offspring_vy = np.random.uniform(-0.5, 0.5, num_offspring) * offspring_velocity_scale

            # Add offspring to cellular type
            ct.x = np.concatenate((ct.x, offspring_x))
            ct.y = np.concatenate((ct.y, offspring_y))
            ct.vx = np.concatenate((ct.vx, offspring_vx))
            ct.vy = np.concatenate((ct.vy, offspring_vy))
            ct.energy = np.concatenate((ct.energy, offspring_energy))
            ct.energy_efficiency = np.concatenate((ct.energy_efficiency, offspring_efficiency))
            ct.speed_factor = np.concatenate((ct.speed_factor, offspring_speed_factor))
            ct.interaction_strength = np.concatenate((ct.interaction_strength, offspring_interaction_strength))
            ct.age = np.concatenate((ct.age, np.zeros(num_offspring)))
            ct.alive = np.concatenate((ct.alive, np.ones(num_offspring, dtype=bool)))
            if ct.mass_based and ct.mass is not None:
                ct.mass = np.concatenate((ct.mass, offspring_mass))


###############################################################
# Interaction Rules
###############################################################

class InteractionRules:
    """
    Manages complex interaction parameters and physics between cellular type pairs.

    Attributes:
    -----------
    config : SimulationConfig
        Simulation configuration.
    rules : List[Tuple[int, int, Dict[str, Any]]]
        Interaction parameters for type pairs.
    give_take_matrix : np.ndarray
        Matrix indicating give-take relationships.
    synergy_matrix : np.ndarray
        Matrix defining synergy factors.
    repulsion_strength : float
        Base strength of overlap repulsion force.
    """
    def __init__(self, config: SimulationConfig, mass_based_type_indices: List[int]):
        """
        Initialize InteractionRules with configuration and mass-based type indices.

        Parameters:
        -----------
        config : SimulationConfig
            Simulation configuration instance.
        mass_based_type_indices : List[int]
            Indices of mass-based types.
        """
        self.config = config
        self.repulsion_strength = 5.0  # Base repulsion strength for overlap prevention

        # Precompute interaction parameters for all type pairs
        self.rules = self._create_interaction_matrix(mass_based_type_indices)
        self.give_take_matrix = self._create_give_take_matrix()
        self.synergy_matrix = self._create_synergy_matrix()

    def _create_interaction_matrix(self, mass_based_type_indices: List[int]) -> List[Tuple[int, int, Dict[str, Any]]]:
        """
        Construct detailed interaction parameters for each type pair.

        Parameters:
        -----------
        mass_based_type_indices : List[int]
            Indices of mass-based types.

        Returns:
        --------
        List[Tuple[int, int, Dict[str, Any]]]
            List of interaction parameters with complex physics rules.
        """
        n_types = self.config.n_cell_types

        # Precompute boolean masks for mass-based and same-type pairs
        mass_based = np.isin(np.arange(n_types), mass_based_type_indices)
        same_type = np.eye(n_types, dtype=bool)

        interaction_rules = []

        # Vectorized parameters
        use_gravity_matrix = mass_based[:, np.newaxis] & mass_based[np.newaxis, :]
        use_gravity_matrix &= np.random.rand(n_types, n_types) < 0.5

        potential_sign_matrix = np.where(np.random.rand(n_types, n_types) < 0.5, -1, 1)
        potential_strength_matrix = np.random.uniform(
            self.config.interaction_strength_range[0],
            self.config.interaction_strength_range[1],
            size=(n_types, n_types)
        ) * potential_sign_matrix

        gravity_factor_matrix = np.where(
            use_gravity_matrix,
            np.random.uniform(0.1, 2.0, size=(n_types, n_types)),
            0.0
        )

        max_dist_matrix = np.random.uniform(50.0, 200.0, size=(n_types, n_types))

        # Advanced interaction parameters
        field_decay_matrix = np.random.uniform(1.0, 2.5, size=(n_types, n_types))
        angular_dependency_matrix = np.random.rand(n_types, n_types) < 0.3
        angular_strength_matrix = np.random.uniform(0.1, 1.0, size=(n_types, n_types))
        elastic_collision_matrix = np.random.rand(n_types, n_types) < 0.4
        restitution_matrix = np.where(same_type, np.random.uniform(0.3, 0.9, size=(n_types, n_types)), 0.5)
        friction_matrix = np.random.uniform(0.0, 0.2, size=(n_types, n_types))
        field_polarization_matrix = np.random.rand(n_types, n_types) < 0.2
        quantum_tunneling_matrix = np.random.rand(n_types, n_types) < 0.1
        tunneling_probability_matrix = np.random.uniform(0.01, 0.1, size=(n_types, n_types))

        # Overlap prevention parameters
        core_repulsion_matrix = np.where(same_type, 10.0, 5.0)
        soft_shell_matrix = np.random.uniform(1.1, 1.5, size=(n_types, n_types)) * self.config.particle_size

        for i in range(n_types):
            for j in range(n_types):
                params: Dict[str, Any] = {
                    "use_potential": True,
                    "use_gravity": use_gravity_matrix[i, j],
                    "potential_strength": potential_strength_matrix[i, j],
                    "gravity_factor": gravity_factor_matrix[i, j],
                    "max_dist": max_dist_matrix[i, j],
                    "field_decay": field_decay_matrix[i, j],
                    "angular_dependency": angular_dependency_matrix[i, j],
                    "angular_strength": angular_strength_matrix[i, j],
                    "elastic_collision": elastic_collision_matrix[i, j],
                    "restitution": restitution_matrix[i, j],
                    "friction": friction_matrix[i, j],
                    "field_polarization": field_polarization_matrix[i, j],
                    "quantum_tunneling": quantum_tunneling_matrix[i, j],
                    "tunneling_probability": tunneling_probability_matrix[i, j],
                    "core_repulsion": core_repulsion_matrix[i, j],
                    "core_range": self.config.particle_size,
                    "soft_shell": soft_shell_matrix[i, j]
                }

                if use_gravity_matrix[i, j]:
                    params.update({
                        "mass_scaling": random.uniform(0.5, 1.5),
                        "inertial_damping": random.uniform(0.1, 0.5),
                        "momentum_transfer": random.uniform(0.7, 1.0)
                    })

                interaction_rules.append((i, j, params))

        return interaction_rules

    def _create_give_take_matrix(self) -> np.ndarray:
        """
        Create an advanced matrix indicating give-take relationships with multiple interaction channels.

        Returns:
        --------
        np.ndarray
            NxN float matrix representing interaction strengths and types.
        """
        n_types = self.config.n_cell_types
        rng = np.random.default_rng()

        indices = np.arange(n_types)
        i_indices, j_indices = np.meshgrid(indices, indices, indexing='ij')
        interaction_prob = 0.1 * (1 + 0.5 * np.cos(i_indices * j_indices * np.pi / n_types))

        random_matrix = rng.random((n_types, n_types))
        mask = (i_indices != j_indices) & (random_matrix < interaction_prob)

        strengths = rng.uniform(0.2, 1.0, size=np.count_nonzero(mask))
        matrix = np.zeros((n_types, n_types), dtype=float)
        matrix[mask] = strengths

        return matrix

    def _create_synergy_matrix(self) -> np.ndarray:
        """
        Create a sophisticated synergy matrix with emergent cooperative behaviors.

        Returns:
        --------
        np.ndarray
            NxN float matrix with complex synergy factors.
        """
        n_types = self.config.n_cell_types
        rng = np.random.default_rng()

        # Initialize synergy matrix
        matrix = np.zeros((n_types, n_types), dtype=float)

        # Vectorized synergy assignment
        synergy_mask = (np.random.rand(n_types, n_types) < 0.15) & (~np.eye(n_types, dtype=bool))
        base_synergy = rng.uniform(0.01, 0.3, size=(n_types, n_types))
        type_compatibility = np.sin(i_indices := np.arange(n_types)[:, None] * np.arange(n_types) * np.pi / (2 * n_types))
        matrix[synergy_mask] = base_synergy[synergy_mask] * (1 + 0.5 * type_compatibility[synergy_mask])

        # Ensure matrix symmetry for stable interactions
        symmetric_matrix = (matrix + matrix.T) / 2
        return symmetric_matrix


###############################################################
# Forces & Interactions
###############################################################

def apply_interaction(
    a_x: np.ndarray,
    a_y: np.ndarray,
    b_x: np.ndarray,
    b_y: np.ndarray,
    params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate force exerted by B on A based on interaction parameters using optimized vectorized operations.

    Parameters:
    -----------
    a_x, a_y : np.ndarray
        Coordinate arrays of A.
    b_x, b_y : np.ndarray
        Coordinate arrays of B.
    params : Dict[str, Any]
        Interaction parameters.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Force component arrays (fx, fy).
    """
    dx = a_x[:, np.newaxis] - b_x
    dy = a_y[:, np.newaxis] - b_y
    d_sq = dx**2 + dy**2
    max_dist_sq = params.get("max_dist", 10.0) ** 2
    mask = (d_sq > 0.0) & (d_sq <= max_dist_sq)

    if not np.any(mask):
        return np.zeros_like(a_x), np.zeros_like(a_y)

    valid_indices, b_indices = np.nonzero(mask)
    valid_dx = dx[mask]
    valid_dy = dy[mask]
    valid_d = np.sqrt(d_sq[mask])

    fx = np.zeros_like(a_x, dtype=np.float64)
    fy = np.zeros_like(a_y, dtype=np.float64)

    if params.get("use_potential", True):
        F_pot = params.get("potential_strength", 1.0) / valid_d
        fx_contrib = F_pot * valid_dx
        fy_contrib = F_pot * valid_dy
        np.add.at(fx, valid_indices, fx_contrib)
        np.add.at(fy, valid_indices, fy_contrib)

    if params.get("use_gravity", False):
        m_a = params.get("m_a", 1.0)
        m_b = params.get("m_b", 1.0)
        gravity_factor = params.get("gravity_factor", 1.0)
        F_grav = gravity_factor * (m_a * m_b) / d_sq[mask]
        fx_contrib = F_grav * valid_dx
        fy_contrib = F_grav * valid_dy
        np.add.at(fx, valid_indices, fx_contrib)
        np.add.at(fy, valid_indices, fy_contrib)

    return fx, fy


def give_take_interaction(
    giver_energy: np.ndarray,
    receiver_energy: np.ndarray,
    giver_mass: Optional[np.ndarray],
    receiver_mass: Optional[np.ndarray],
    config: SimulationConfig
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Facilitate energy and mass transfer from givers to receivers using optimized vectorized operations.

    Parameters:
    -----------
    giver_energy : np.ndarray
        Energy array of givers.
    receiver_energy : np.ndarray
        Energy array of receivers.
    giver_mass : Optional[np.ndarray]
        Mass array of givers.
    receiver_mass : Optional[np.ndarray]
        Mass array of receivers.
    config : SimulationConfig
        Simulation configuration instance.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
        Updated energy and mass arrays.
    """
    transfer_factor = config.energy_transfer_factor
    transfer_amount = receiver_energy * transfer_factor
    receiver_energy = np.subtract(receiver_energy, transfer_amount, out=receiver_energy, where=transfer_amount != 0)
    np.add(giver_energy, transfer_amount, out=giver_energy)

    if config.mass_transfer and giver_mass is not None and receiver_mass is not None:
        mass_transfer = receiver_mass * transfer_factor
        receiver_mass = np.subtract(receiver_mass, mass_transfer, out=receiver_mass, where=mass_transfer != 0)
        np.add(giver_mass, mass_transfer, out=giver_mass)

    return giver_energy, receiver_energy, giver_mass, receiver_mass


def apply_clustering(ct: CellularTypeData, config: SimulationConfig) -> None:
    """
    Apply clustering forces for flocking behavior within a cellular type,
    considering only neighbors in a sector in front of the component's
    direction of motion.

    Parameters:
    -----------
    ct : CellularTypeData
        Cellular type data.
    config : SimulationConfig
        Simulation configuration instance.
    """
    n = ct.x.size
    if n < 2:
        return

    points = np.column_stack((ct.x, ct.y))
    tree = cKDTree(points)
    neighbors_list = tree.query_ball_point(points, r=config.cluster_radius)

    dvx = np.zeros(n, dtype=np.float64)
    dvy = np.zeros(n, dtype=np.float64)
    alive = ct.alive

    directions = np.stack((ct.vx, ct.vy), axis=1)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    with np.errstate(invalid='ignore'):
        directions = np.divide(directions, norms, out=np.zeros_like(directions), where=norms > 1e-10)

    for idx, neighbors in enumerate(neighbors_list):
        if not alive[idx]:
            continue

        active_neighbors = np.array(neighbors)
        active_neighbors = active_neighbors[(active_neighbors != idx) & alive[active_neighbors]]
        if active_neighbors.size == 0:
            continue

        neighbor_vectors = points[active_neighbors] - points[idx]
        vector_norms = np.linalg.norm(neighbor_vectors, axis=1)
        valid = vector_norms > 1e-10
        if not np.any(valid):
            continue

        neighbor_vectors = neighbor_vectors[valid]
        vector_norms = vector_norms[valid]
        directions_idx = directions[idx]

        dot_products = np.dot(neighbor_vectors, directions_idx)
        angles = np.arccos(np.clip(dot_products / vector_norms, -1.0, 1.0))
        sector_mask = angles <= (np.pi / 3)

        filtered_neighbors = active_neighbors[valid][sector_mask]
        if filtered_neighbors.size == 0:
            continue

        neighbor_v = directions[filtered_neighbors]
        avg_vx, avg_vy = neighbor_v[:, 0].mean(), neighbor_v[:, 1].mean()
        center = points[filtered_neighbors].mean(axis=0)
        separation = points[idx] - points[filtered_neighbors].mean(axis=0)

        dvx[idx] += (
            config.alignment_strength * (avg_vx - ct.vx[idx]) +
            config.cohesion_strength * (center[0] - ct.x[idx]) +
            config.separation_strength * separation[0]
        )
        dvy[idx] += (
            config.alignment_strength * (avg_vy - ct.vy[idx]) +
            config.cohesion_strength * (center[1] - ct.y[idx]) +
            config.separation_strength * separation[1]
        )

    ct.vx = np.add(ct.vx, dvx)
    ct.vy = np.add(ct.vy, dvy)


def apply_synergy(
    energyA: np.ndarray,
    energyB: np.ndarray,
    synergy_factor: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance energy between two allied components based on synergy factor using optimized vectorized operations.

    Parameters:
    -----------
    energyA : np.ndarray
        Energy array of component A.
    energyB : np.ndarray
        Energy array of component B.
    synergy_factor : float
        Energy sharing proportion.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Updated energy arrays for A and B.
    """
    energyA, energyB = np.asarray(energyA), np.asarray(energyB)

    if energyA.shape != energyB.shape:
        raise ValueError("energyA and energyB must have the same shape.")

    avg_energy = (energyA + energyB) * 0.5
    np.multiply(energyA, (1.0 - synergy_factor), out=energyA)
    np.multiply(energyB, (1.0 - synergy_factor), out=energyB)
    np.add(energyA, avg_energy * synergy_factor, out=energyA)
    np.add(energyB, avg_energy * synergy_factor, out=energyB)

    return energyA, energyB


def apply_colony_energy_sharing(ct: CellularTypeData, config: SimulationConfig) -> None:
    """
    Distribute energy among nearby components within a colony using optimized vectorized operations
    and a KDTree for efficient neighbor searching.

    Parameters:
    -----------
    ct : CellularTypeData
        Cellular type data.
    config : SimulationConfig
        Simulation configuration instance.
    """
    n = ct.x.size
    if n < 2:
        return

    points = np.column_stack((ct.x, ct.y))
    tree = cKDTree(points)
    sharing_radius = config.synergy_range
    neighbors_list = tree.query_ball_point(points, r=sharing_radius)

    energy_changes = np.zeros(n, dtype=np.float64)
    synergy_matrix = config.synergy_matrix

    for idx, neighbors in enumerate(neighbors_list):
        neighbors = np.array(neighbors)
        higher_indices = neighbors[neighbors > idx]
        if higher_indices.size == 0:
            continue

        avg_energy = (ct.energy[idx] + ct.energy[higher_indices]) * 0.5
        energy_diff = avg_energy - ct.energy[idx]
        factors = synergy_matrix[idx, higher_indices]

        energy_changes[idx] += np.sum(energy_diff * factors)
        np.add.at(energy_changes, higher_indices, -energy_diff * factors)

    np.add(ct.energy, energy_changes, out=ct.energy)
    np.clip(ct.energy, 0.0, None, out=ct.energy)

###############################################################
# Environmental Manager
###############################################################

class EnvironmentalManager:
    """
    Manages environmental zones and their effects on cellular components.

    Attributes:
    -----------
    zones : List[Dict[str, Any]]
        Environmental zones with absolute positions and precomputed radii squared.
    """
    def __init__(self, config: SimulationConfig, screen_width: int, screen_height: int):
        """
        Initialize EnvironmentalManager with zones and screen dimensions.

        Parameters:
        -----------
        config : SimulationConfig
            Simulation configuration instance.
        screen_width, screen_height : int
            Window dimensions in pixels.
        """
        self.zones = config.zones
        for zone in self.zones:
            zone["x_abs"] = zone["x"] * screen_width
            zone["y_abs"] = zone["y"] * screen_height
            zone["radius_sq"] = zone["radius"] ** 2

    def apply_zone_effects(self, ct: CellularTypeData) -> None:
        """
        Modify energy levels based on presence within zones using cKDTree for efficiency.

        Parameters:
        -----------
        ct : CellularTypeData
            Cellular type data.
        """
        if not self.zones:
            return

        component_coords = np.column_stack((ct.x, ct.y))
        tree = cKDTree(component_coords)

        for zone in self.zones:
            indices = tree.query_ball_point((zone["x_abs"], zone["y_abs"]), r=zone["radius"])
            ct.energy[indices] *= zone["effect"].get("energy_modifier", 1.0)

###############################################################
# Renderer Class
###############################################################

class Renderer:
    """
    Handles rendering of components, plants, and food particles using Pygame surfaces.

    Attributes:
    -----------
    surface : pygame.Surface
        Main Pygame surface.
    config : SimulationConfig
        Simulation configuration.
    color_map : Dict[str, Tuple[int, int, int]]
        Mapping of color names to RGB tuples.
    precomputed_colors : Dict[str, Tuple[int, int, int]]
        Precomputed color values.
    particle_surface : pygame.Surface
        Off-screen surface for particles.
    font : pygame.font.Font
        Font for FPS display.
    """
    def __init__(self, surface: pygame.Surface, config: SimulationConfig):
        """
        Initialize Renderer with main surface and configuration.

        Parameters:
        -----------
        surface : pygame.Surface
            Main Pygame surface.
        config : SimulationConfig
            Simulation configuration instance.
        """
        self.surface = surface
        self.config = config

        self.color_map: Dict[str, Tuple[int, int, int]] = {
            "neon_red": (255, 20, 147),
            "neon_green": (57, 255, 20),
            "neon_blue": (0, 191, 255),
            "neon_yellow": (255, 255, 0),
            "neon_cyan": (0, 255, 255),
            "neon_magenta": (255, 0, 255),
            "neon_orange": (255, 165, 0),
            "neon_lime": (50, 205, 50),
            "neon_purple": (148, 0, 211),
            "neon_pink": (255, 105, 180),
            "neon_gold": (255, 215, 0),
            "neon_aqua": (127, 255, 212),
            "neon_chartreuse": (127, 255, 0),
            "neon_crimson": (220, 20, 60),
            "neon_navy": (0, 0, 128),
            "neon_maroon": (128, 0, 0),
            "neon_fuchsia": (255, 0, 255),
            "neon_teal": (0, 128, 128),
            "neon_electric_blue": (125, 249, 255),
            "neon_bright_green": (102, 255, 0),
            "neon_spring_green": (0, 255, 127),
            "neon_sky_blue": (135, 206, 235),
            "neon_pale_green": (152, 251, 152),
            "neon_dark_purple": (148, 0, 211)
        }

        # Precompute color array for faster access
        self.precomputed_colors = {name: rgb for name, rgb in self.color_map.items()}

        # Initialize particle surface for double buffering
        self.particle_surface = pygame.Surface(self.surface.get_size(), flags=pygame.SRCALPHA)
        self.particle_surface = self.particle_surface.convert_alpha()

        # Initialize font for FPS display
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", self.config.font_size)

    def handle_resize(self, new_size: Tuple[int, int]) -> None:
        """
        Handle window resize events by adjusting the particle surface.

        Parameters:
        -----------
        new_size : Tuple[int, int]
            New window size (width, height).
        """
        self.surface = pygame.display.set_mode(new_size, pygame.RESIZABLE)
        self.particle_surface = pygame.Surface(self.surface.get_size(), flags=pygame.SRCALPHA)
        self.particle_surface = self.particle_surface.convert_alpha()

    def draw_components(self, ct: CellularTypeData) -> None:
        """
        Render all alive components of a cellular type using batch drawing for efficiency.

        Parameters:
        -----------
        ct : CellularTypeData
            Cellular type data.
        """
        alive_indices = np.where(ct.alive)[0]
        if alive_indices.size == 0:
            return

        # Extract positions and attributes
        x = ct.x[alive_indices].astype(int)
        y = ct.y[alive_indices].astype(int)
        energy = ct.energy[alive_indices]
        speed_factor = ct.speed_factor[alive_indices]
        color_name = ct.color

        # Retrieve base color
        base_color = np.array(self.precomputed_colors.get(color_name, (255, 0, 255)), dtype=float)

        # Compute color intensity based on energy and speed factor
        health = np.clip(energy, 0.0, 100.0)
        intensity = np.clip(health / 100.0, 0.0, 1.0)
        final_colors = base_color * intensity[:, np.newaxis] * speed_factor[:, np.newaxis] + (1 - intensity[:, np.newaxis]) * 100
        final_colors = np.clip(final_colors, 0, 255).astype(int)

        # Optimized batch drawing
        particle_size = int(self.config.particle_size)
        for xi, yi, color in zip(x, y, final_colors):
            color_tuple = tuple(color)
            pygame.gfxdraw.filled_circle(self.particle_surface, xi, yi, particle_size, color_tuple)
            pygame.gfxdraw.aacircle(self.particle_surface, xi, yi, particle_size, color_tuple)

    def render_background(self, is_day: bool) -> None:
        """
        Fill background based on day or night.

        Parameters:
        -----------
        is_day : bool
            Daytime status.
        """
        background_color = (135, 206, 235) if is_day else (25, 25, 112)
        self.surface.fill(background_color)

    def render_fps(self, fps: float) -> None:
        """
        Render FPS on the screen.

        Parameters:
        -----------
        fps : float
            Current frames per second.
        """
        fps_text = f"FPS: {fps:.2f}"
        text_surface = self.font.render(fps_text, True, (255, 255, 255))
        self.particle_surface.blit(text_surface, (10, 10))

    def render(self, is_day: bool, fps: float) -> None:
        """
        Composite rendered elements onto the main surface.

        Parameters:
        -----------
        is_day : bool
            Daytime status.
        fps : float
            Current FPS.
        """
        self.render_background(is_day)
        self.render_fps(fps)
        self.surface.blit(self.particle_surface, (0, 0))
        self.particle_surface.fill((0, 0, 0, 0))  # Clear particle surface for next frame

###############################################################
# Cellular Automata (Main Simulation)
###############################################################

class CellularAutomata:
    """
    Core simulation class managing initialization, updating, rendering, and lifecycle.

    Attributes:
    -----------
    config : SimulationConfig
        Simulation configuration.
    screen : pygame.Surface
        Main Pygame surface.
    clock : pygame.time.Clock
        Clock for managing FPS.
    frame_count : int
        Current frame count.
    run_flag : bool
        Simulation running status.
    edge_buffer : float
        Buffer distance from window edges.
    colors : List[str]
        Color names for cellular types.
    type_manager : CellularTypeManager
        Manager handling cellular types.
    rules_manager : InteractionRules
        Manager handling interaction rules.
    env_manager : EnvironmentalManager
        Manager handling environmental zones.
    plant_manager : PlantManager
        Manager handling plants.
    renderer : Renderer
        Renderer instance.
    day_night_cycle : Dict[str, Any]
        Day-night cycle state.
    protected_colonies : List[int]
        Indices of protected colonies.
    paused : bool
        Simulation paused status.
    pause_start_time : int
        Timestamp when pause started.
    cull_factor : float
        Dynamic culling factor.
    """
    def __init__(self, config: SimulationConfig):
        """
        Initialize CellularAutomata with configuration.

        Parameters:
        -----------
        config : SimulationConfig
            Simulation configuration instance.
        """
        self.config = config
        pygame.init()

        screen_width, screen_height = 800, 600  # Default window size
        # Initialize window with RESIZABLE flag and standard window decorations
        self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Ultra-Advanced Emergent Cellular Automata Simulation")

        self.clock = pygame.time.Clock()
        self.frame_count = 0
        self.run_flag = True

        self.edge_buffer = 0.01 * max(screen_width, screen_height)

        self.colors = generate_vibrant_colors(self.config.n_cell_types)
        n_mass_types = int(self.config.mass_based_fraction * self.config.n_cell_types)
        mass_based_type_indices = list(range(n_mass_types))

        self.type_manager = CellularTypeManager(self.config, self.colors, mass_based_type_indices)

        # Initialize cellular types
        for i in range(self.config.n_cell_types):
            if i < n_mass_types:
                mass_val = random.uniform(*self.config.mass_range)
                ct = CellularTypeData(
                    type_id=i,
                    color=self.colors[i],
                    n_particles=self.config.particles_per_type,
                    window_width=screen_width,
                    window_height=screen_height,
                    config=self.config,
                    mass=mass_val,
                    base_velocity_scale=self.config.base_velocity_scale
                )
            else:
                ct = CellularTypeData(
                    type_id=i,
                    color=self.colors[i],
                    n_particles=self.config.particles_per_type,
                    window_width=screen_width,
                    window_height=screen_height,
                    config=self.config,
                    mass=None,
                    base_velocity_scale=self.config.base_velocity_scale
                )
            self.type_manager.add_cellular_type_data(ct)

        self.rules_manager = InteractionRules(self.config, mass_based_type_indices)
        self.env_manager = EnvironmentalManager(self.config, screen_width, screen_height)
        
        self.renderer = Renderer(self.screen, self.config)

        self.paused = True
        self.pause_start_time = pygame.time.get_ticks()
        logging.info("Simulation initialized and paused for initial setup.")
        self.cull_factor: float = 1.0  # Initial culling factor

    def main_loop(self) -> None:
        """
        Execute the main simulation loop, handling events, updates, and rendering.
        """
        fps_history = []
        fps_history_length = 60

        while self.run_flag:
            self.frame_count += 1
            if self.config.max_frames > 0 and self.frame_count > self.config.max_frames:
                self.run_flag = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run_flag = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.run_flag = False
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    self.renderer.handle_resize(event.size)
                    self.env_manager.zones = self.adjust_zones(event.size)

            current_time = pygame.time.get_ticks()
            elapsed_time = current_time - self.pause_start_time

            if self.paused and elapsed_time >= self.config.initial_pause_duration:
                self.paused = False
                logging.info("Initial setup complete. Simulation is now running.")

            if not self.paused:
                self.apply_all_interactions()
                for ct in self.type_manager.cellular_types:
                    self.env_manager.apply_zone_effects(ct)
                for ct in self.type_manager.cellular_types:
                    apply_clustering(ct, self.config)
                self.type_manager.reproduce()
                self.type_manager.remove_dead_in_all_types()
                for ct in self.type_manager.cellular_types:
                    self.renderer.draw_components(ct)
                self.handle_boundary_reflections()

            current_fps = self.clock.get_fps()
            fps_history.append(current_fps)
            if len(fps_history) > fps_history_length:
                fps_history.pop(0)
            average_fps = np.mean(fps_history) if fps_history else current_fps

            self.renderer.render(False, average_fps)
            pygame.display.flip()
            self.clock.tick(60)  # Standardized to 60 FPS for consistency

            if not self.paused:
                self.adaptive_intelligent_culling(average_fps=average_fps)

    def adjust_zones(self, new_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Adjust environmental zones based on the new window size.

        Parameters:
        -----------
        new_size : Tuple[int, int]
            New window size (width, height).

        Returns:
        --------
        List[Dict[str, Any]]
            Updated environmental zones.
        """
        updated_zones = []
        for zone in self.config.zones:
            updated_zone = zone.copy()
            updated_zone["x_abs"] = zone["x"] * new_size[0]
            updated_zone["y_abs"] = zone["y"] * new_size[1]
            updated_zone["radius_sq"] = zone["radius"] ** 2
            updated_zones.append(updated_zone)
        return updated_zones

    def apply_all_interactions(self) -> None:
        """
        Apply all defined interactions between cellular types.
        """
        for (i, j, params) in self.rules_manager.rules:
            self.apply_interaction_between_types(i, j, params)

    def apply_interaction_between_types(self, i: int, j: int, params: Dict[str, Any]) -> None:
        """
        Apply interaction parameters between two cellular types using cKDTree for efficiency.

        Parameters:
        -----------
        i, j : int
            Type indices.
        params : Dict[str, Any]
            Interaction parameters.
        """
        ct_i = self.type_manager.get_cellular_type_by_id(i)
        ct_j = self.type_manager.get_cellular_type_by_id(j)

        synergy_factor = self.rules_manager.synergy_matrix[i, j]
        is_giver = self.rules_manager.give_take_matrix[i, j]

        n_i, n_j = ct_i.x.size, ct_j.x.size
        if n_i == 0 or n_j == 0:
            return

        use_gravity = params.get("use_gravity", False)
        if use_gravity and ct_i.mass_based and ct_j.mass_based and ct_i.mass is not None and ct_j.mass is not None:
            m_a = ct_i.mass
            m_b = ct_j.mass
        else:
            use_gravity = False

        # Use cKDTree for efficient neighbor search
        tree_i = cKDTree(np.column_stack((ct_i.x, ct_i.y)))
        tree_j = cKDTree(np.column_stack((ct_j.x, ct_j.y)))
        pairs = tree_i.query_ball_tree(tree_j, r=params["max_dist"])

        # Convert pairs to numpy arrays for vectorized operations
        i_indices = np.array([i for i, neighbors in enumerate(pairs) for _ in neighbors], dtype=np.int32)
        j_indices = np.array([j for neighbors in pairs for j in neighbors], dtype=np.int32)

        if i_indices.size == 0:
            return

        # Calculate distances and forces using vectorized operations
        dx_arr = ct_i.x[i_indices] - ct_j.x[j_indices]
        dy_arr = ct_i.y[i_indices] - ct_j.y[j_indices]
        dist_sq_arr = dx_arr * dx_arr + dy_arr * dy_arr

        # Filter out particles beyond max distance
        mask = dist_sq_arr <= params["max_dist"] ** 2
        if not np.any(mask):
            return

        i_indices = i_indices[mask]
        j_indices = j_indices[mask]
        dx_arr = dx_arr[mask]
        dy_arr = dy_arr[mask]
        dist_sq_arr = dist_sq_arr[mask]
        dist_arr = np.sqrt(dist_sq_arr)

        # Calculate forces using vectorized operations
        fx = np.zeros_like(dist_arr)
        fy = np.zeros_like(dist_arr)

        use_potential = params.get("use_potential", True)
        if use_potential:
            F_pot = params.get("potential_strength", 1.0) / np.maximum(dist_arr, 1e-6)  # Avoid division by zero
            fx += F_pot * dx_arr
            fy += F_pot * dy_arr

        if use_gravity:
            F_grav = params.get("gravity_factor", 1.0) * (m_a[i_indices] * m_b[j_indices]) / np.maximum(dist_sq_arr, 1e-6)
            fx += F_grav * dx_arr
            fy += F_grav * dy_arr

        # Aggregate forces per unique idx_i using np.add.at
        np.add.at(ct_i.vx, i_indices, fx)
        np.add.at(ct_i.vy, i_indices, fy)

        # Handle give-take interactions
        if is_giver and self.config.enable_culling:
            give_take_mask = dist_sq_arr <= self.config.predation_range ** 2
            if np.any(give_take_mask):
                giver_indices = i_indices[give_take_mask]
                receiver_indices = j_indices[give_take_mask]
                giver_energy = ct_i.energy[giver_indices]
                receiver_energy = ct_j.energy[receiver_indices]
                giver_mass = ct_i.mass[giver_indices] if ct_i.mass_based else None
                receiver_mass = ct_j.mass[receiver_indices] if ct_j.mass_based else None

                updated = give_take_interaction(
                    giver_energy,
                    receiver_energy,
                    giver_mass,
                    receiver_mass,
                    self.config
                )
                ct_i.energy[giver_indices] = updated[0]
                ct_j.energy[receiver_indices] = updated[1]

                if ct_i.mass_based and ct_i.mass is not None and updated[2] is not None:
                    ct_i.mass[giver_indices] = updated[2]
                if ct_j.mass_based and ct_j.mass is not None and updated[3] is not None:
                    ct_j.mass[receiver_indices] = updated[3]

        # Handle synergy interactions
        if synergy_factor > 0.0 and self.config.synergy_range > 0.0:
            synergy_mask = dist_sq_arr <= self.config.synergy_range ** 2
            if np.any(synergy_mask):
                synergy_indices_i = i_indices[synergy_mask]
                synergy_indices_j = j_indices[synergy_mask]
                energyA = ct_i.energy[synergy_indices_i]
                energyB = ct_j.energy[synergy_indices_j]
                new_energyA, new_energyB = apply_synergy(energyA, energyB, synergy_factor)
                ct_i.energy[synergy_indices_i] = new_energyA
                ct_j.energy[synergy_indices_j] = new_energyB

        # Apply friction and global temperature in a vectorized manner
        ct_i.vx *= self.config.friction
        ct_i.vy *= self.config.friction
        ct_i.vx += np.random.uniform(-0.5, 0.5, n_i) * self.config.global_temperature
        ct_i.vy += np.random.uniform(-0.5, 0.5, n_i) * self.config.global_temperature

        # Update positions
        ct_i.x += ct_i.vx
        ct_i.y += ct_i.vy

        # Handle boundary reflections
        self.handle_boundary_reflection(ct_i)

        # Update age and alive status
        ct_i.age_components()
        ct_i.update_states()
        ct_i.update_alive()

    def handle_boundary_reflections(self) -> None:
        """
        Reflect cellular components off window boundaries.
        """
        for ct in self.type_manager.cellular_types:
            if ct.x.size == 0:
                continue

            # Reflect off left and right edges
            left = self.config.edge_buffer
            right = self.screen.get_width() - self.config.edge_buffer
            close_left = ct.x < left
            close_right = ct.x > right
            ct.vx[close_left | close_right] *= -1
            ct.x = np.clip(ct.x, left, right)

            # Reflect off top and bottom edges
            top = self.config.edge_buffer
            bottom = self.screen.get_height() - self.config.edge_buffer
            close_top = ct.y < top
            close_bottom = ct.y > bottom
            ct.vy[close_top | close_bottom] *= -1
            ct.y = np.clip(ct.y, top, bottom)

    def handle_boundary_reflection(self, ct: CellularTypeData) -> None:
        """
        Reflect cellular components of a single type off window boundaries, ensuring they stay away
        from the boundaries and do not stick to them. This implementation is simple and efficient,
        using vectorized operations for high performance.

        Parameters:
        -----------
        ct : CellularTypeData
            Cellular type data.
        """
        if ct.x.size == 0:
            return

        # Define boundary limits
        left = self.config.edge_buffer
        right = self.screen.get_width() - self.config.edge_buffer
        top = self.config.edge_buffer
        bottom = self.screen.get_height() - self.config.edge_buffer

        # Define a safe margin inside the boundaries
        margin = 10.0

        # Handle horizontal boundaries (left and right)
        close_left = ct.x < left + margin
        close_right = ct.x > right - margin

        # Reflect velocity and reposition particles
        ct.vx[close_left] = np.abs(ct.vx[close_left])
        ct.vx[close_right] = -np.abs(ct.vx[close_right])

        # Reposition particles to stay within the margin
        ct.x[close_left] = left + margin
        ct.x[close_right] = right - margin

        # Handle vertical boundaries (top and bottom)
        close_top = ct.y < top + margin
        close_bottom = ct.y > bottom - margin

        # Reflect velocity and reposition particles
        ct.vy[close_top] = np.abs(ct.vy[close_top])
        ct.vy[close_bottom] = -np.abs(ct.vy[close_bottom])

        # Reposition particles to stay within the margin
        ct.y[close_top] = top + margin
        ct.y[close_bottom] = bottom - margin
    def adaptive_intelligent_culling(self, average_fps: float) -> None:
        """
        Advanced selective culling based on multi-factor fitness evaluation and FPS.
        Uses highly optimized vectorized operations, adaptive thresholds, and self-correcting mechanisms.
        Encourages complex emergent behaviors while maintaining performance and stability.
        Enhances trait and type diversity to prevent the complete removal of any specific type.
        """
        if not self.config.enable_culling:
            return

        try:
            # Use exponential moving average for FPS smoothing
            alpha = 0.2
            self.smoothed_fps = getattr(self, 'smoothed_fps', average_fps)
            self.smoothed_fps = alpha * average_fps + (1 - alpha) * self.smoothed_fps
            current_fps = self.smoothed_fps

            # Get total particles with error checking
            total_particles = np.sum([
                ct.x.size if hasattr(ct, 'x') and ct.x is not None and ct.x.size > 0 else 0
                for ct in self.type_manager.cellular_types
            ], dtype=np.int32)

            if total_particles <= max(self.config.min_total_particles, 100):
                return

            # Compute type frequencies to promote diversity
            type_counts = np.array([
                ct.x.size for ct in self.type_manager.cellular_types
                if hasattr(ct, 'x') and ct.x is not None and ct.x.size > 0
            ], dtype=np.int32)

            if len(type_counts) == 0:
                return

            type_frequencies = type_counts / total_particles
            type_frequencies = np.where(type_frequencies > 0, type_frequencies, 1.0)

            # Adaptive culling parameters with self-correction
            self.cull_history = getattr(self, 'cull_history', [])
            self.cull_history.append(current_fps)
            if len(self.cull_history) > 120:
                self.cull_history.pop(0)

            # Use robust linear fit
            mask = np.isfinite(self.cull_history)
            if np.sum(mask) > 1:
                fps_trend = np.polyfit(np.arange(len(self.cull_history))[mask],
                                     np.array(self.cull_history)[mask], 1)[0]
            else:
                fps_trend = 0.0

            # Dynamic culling ratio with trend analysis
            base_cull_ratio = np.float32(self.config.culling_ratio)
            if current_fps < self.config.fps_low_threshold:
                fps_deficit = (self.config.fps_low_threshold - current_fps) / self.config.fps_low_threshold
                trend_factor = 1.0 + (0.5 if fps_trend < 0 else -0.2)
                cull_ratio = np.clip(base_cull_ratio * (1.0 + fps_deficit) * trend_factor, 0.0, 0.5)
                self.cull_factor = np.clip(getattr(self, 'cull_factor', 1.0) * (1.1 + fps_deficit * 0.1), 1.0, 5.0)
            else:
                cull_ratio = base_cull_ratio
                self.cull_factor = max(1.0, getattr(self, 'cull_factor', 1.0) * 0.95)

            # Calculate number to cull while preserving minimum particles per type
            min_particles_per_type = max(1, int(0.05 * self.config.min_total_particles))
            num_to_cull = np.int32(total_particles * cull_ratio * self.cull_factor)
            max_cullable = total_particles - (min_particles_per_type * len(self.type_manager.cellular_types))
            num_to_cull = min(num_to_cull, max_cullable)

            if num_to_cull <= 0:
                return

            # Dynamic analysis parameters
            performance_scale = np.clip(current_fps / self.config.fps_low_threshold, 0.5, 2.0)
            radii = np.array([50.0, 25.0, 12.5], dtype=np.float32) * performance_scale

            # Screen boundaries
            screen_width = max(1, self.screen.get_width())
            screen_height = max(1, self.screen.get_height())
            edge_buffer = min(self.config.edge_buffer, min(screen_width, screen_height) * 0.1)

            left = edge_buffer
            right = screen_width - edge_buffer
            top = edge_buffer
            bottom = screen_height - edge_buffer

            type_metrics = []
            type_frequencies_iter = iter(type_frequencies)

            for ct in self.type_manager.cellular_types:
                if not hasattr(ct, 'x') or ct.x is None or ct.x.size == 0 or not hasattr(ct, 'y'):
                    continue

                try:
                    # Validate coordinates
                    valid_mask = (
                        np.isfinite(ct.x) & np.isfinite(ct.y) &
                        (ct.x >= 0) & (ct.x <= screen_width) &
                        (ct.y >= 0) & (ct.y <= screen_height)
                    )

                    if not np.any(valid_mask):
                        next(type_frequencies_iter, None)
                        continue

                    points = np.column_stack((
                        np.clip(ct.x[valid_mask], left, right),
                        np.clip(ct.y[valid_mask], top, bottom)
                    )).astype(np.float32)

                    if points.size == 0:
                        next(type_frequencies_iter, None)
                        continue

                    tree = cKDTree(points, leafsize=32)

                    # Border analysis
                    dx = np.minimum(points[:, 0] - left, right - points[:, 0])
                    dy = np.minimum(points[:, 1] - top, bottom - points[:, 1])
                    border_dist = np.minimum(dx, dy)
                    border_penalty = np.exp(-border_dist / (edge_buffer * 0.5))

                    # Multi-scale clustering
                    clustering_metrics = np.zeros((len(radii), points.shape[0]), dtype=np.float32)
                    for i, radius in enumerate(radii):
                        neighbors = tree.query_ball_point(points, radius)
                        clustering_metrics[i] = np.array([len(n) for n in neighbors], dtype=np.float32)

                    # Normalize clustering metrics
                    min_vals = clustering_metrics.min(axis=1, keepdims=True)
                    range_vals = np.maximum(clustering_metrics.max(axis=1, keepdims=True) - min_vals, 1e-10)
                    clustering_metrics = (clustering_metrics - min_vals) / range_vals
                    clustering_metrics = np.nan_to_num(clustering_metrics, nan=0.0)

                    clustering_score = clustering_metrics.mean(axis=0)

                    # Spatial organization
                    k = min(5, points.shape[0] - 1)
                    if k > 0:
                        distances = tree.query(points, k=k+1, workers=-1)[0][:, 1:]
                        spacing_score = 1.0 - np.exp(-distances / radii[0])
                        spacing_score = spacing_score.mean(axis=1)
                    else:
                        spacing_score = np.ones(points.shape[0], dtype=np.float32)

                    # Calculate fitness
                    fitness = np.zeros(points.shape[0], dtype=np.float32)
                    fitness_attrs = ['energy', 'age', 'speed_factor', 'interaction_strength']
                    for attr in fitness_attrs:
                        if hasattr(ct, attr):
                            values = getattr(ct, attr)[valid_mask]
                            if values.size > 0:
                                min_val = values.min()
                                range_val = np.maximum(values.max() - min_val, 1e-10)
                                normalized = (values - min_val) / range_val
                                fitness += np.float32(self.config.culling_fitness_weights.get(f"{attr}_weight", 0.0)) * np.nan_to_num(normalized, nan=0.0)

                    # Add complexity components
                    complexity_weights = {
                        'clustering': 0.3,
                        'spacing': 0.2,
                        'border': -0.4
                    }

                    fitness += (
                        complexity_weights['clustering'] * clustering_score +
                        complexity_weights['spacing'] * spacing_score +
                        complexity_weights['border'] * border_penalty
                    )

                    # Incorporate type diversity
                    type_freq = next(type_frequencies_iter, 1.0)
                    fitness *= (1.0 + (1.0 - type_freq))

                    # Normalize fitness
                    fitness = np.nan_to_num(fitness, nan=-np.inf)
                    min_fit = fitness.min()
                    range_fit = np.maximum(fitness.max() - min_fit, 1e-10)
                    fitness = (fitness - min_fit) / range_fit

                    type_metrics.append((ct, fitness, tree, points, valid_mask, type_freq))

                except Exception as e:
                    logging.warning(f"Error processing cellular type: {str(e)}")
                    continue

            # Perform culling
            if not type_metrics:
                return

            all_fitness = np.concatenate([metrics[1] for metrics in type_metrics if metrics[1].size > 0])
            if all_fitness.size == 0:
                return

            threshold = np.percentile(all_fitness, min(95, int(cull_ratio * 100 * self.cull_factor)))

            culled_count = 0
            for ct, fitness, tree, points, valid_mask, type_freq in type_metrics:
                if culled_count >= num_to_cull:
                    break

                min_keep = max(min_particles_per_type, int(type_freq * self.config.min_total_particles))
                current_type_count = np.sum(valid_mask)
                available_to_cull = current_type_count - min_keep
                
                if available_to_cull <= 0:
                    continue

                cull_mask = fitness < threshold
                num_cull_candidates = np.sum(cull_mask)
                max_cull = min(num_to_cull - culled_count, num_cull_candidates, available_to_cull)
                
                if max_cull <= 0:
                    continue

                cull_indices = np.where(cull_mask)[0][:max_cull]
                if cull_indices.size == 0:
                    continue

                full_survival_mask = np.ones(ct.x.size, dtype=bool)
                valid_indices = np.where(valid_mask)[0]
                full_survival_mask[valid_indices[cull_indices]] = False

                # Update attributes
                attrs = ['x', 'y', 'vx', 'vy', 'energy', 'alive', 'age',
                        'energy_efficiency', 'speed_factor', 'interaction_strength', 'mass']
                for attr in attrs:
                    if hasattr(ct, attr):
                        try:
                            attr_data = getattr(ct, attr)
                            if isinstance(attr_data, np.ndarray):
                                setattr(ct, attr, attr_data[full_survival_mask])
                        except Exception as e:
                            logging.warning(f"Error updating attribute {attr}: {str(e)}")
                            continue

                culled_count += len(cull_indices)

            # Update culling effectiveness
            if culled_count == 0:
                self.cull_factor = max(1.0, self.cull_factor * 0.9)

        except Exception as e:
            logging.error(f"Critical error in adaptive_intelligent_culling: {str(e)}")
            self.cull_factor = 1.0

    def run(self) -> None:
        """
        Start and manage the simulation loop.
        """
        self.main_loop()

###############################################################
# Entry Point
###############################################################

def main():
    """
    Entry point for the simulation.
    Initializes configuration and starts the simulation.
    """
    config = SimulationConfig()
    cellular_automata = CellularAutomata(config)
    cellular_automata.run()

if __name__ == "__main__":
    main()
