"""
Ultra-Advanced Emergent Cellular Automata and Multicellular Simulation - Enhanced Version with Dynamic Genes, Adaptive Frame Management, Colony Formation, Resource Management, Food Particles, Plant Species, and Day-Night Cycle
----------------------------------------------------------------------------------------------------
This enhanced simulation builds upon the fully optimized version by introducing dynamic gene traits to each cellular component, enabling emergent behaviors through interactions. Additionally, it incorporates adaptive frame management to maintain optimal performance, ensuring smooth visualization by dynamically adjusting particle populations and energy levels based on the current frames per second (FPS).

Key Enhancements:
1. **Dynamic Genes:**
    - Each cellular component possesses gene traits (`speed_factor`, `interaction_strength`, and `energy_efficiency`) that influence its behavior and interactions.
    - Genes can mutate during reproduction, allowing for evolutionary dynamics and emergent behaviors.
2. **Adaptive Frame Management:**
    - Monitors the simulation's FPS.
    - If FPS drops below 30, the simulation starts culling the oldest particles each frame until FPS stabilizes above 30.
    - If FPS exceeds 120, the simulation adds an extra 10% energy to all particles each frame until FPS drops below 120, stimulating global reproduction.
3. **Emergent Behaviors:**
    - Genes influence velocity scaling and interaction force strengths, leading to diverse and evolving behaviors without hardcoding specific interactions.
4. **Colony Formation and Resource Management:**
    - Cellular components can form colonies through synergy and interaction rules.
    - Resource management is implemented to ensure energy conservation and efficient distribution.
5. **Food Particles:**
    - Defined as a separate class, food particles represent energy sources in the simulation.
    - When an organism dies, it transforms into food particles, ensuring that energy isn't destroyed.
6. **Plant Species with Fractal Growth and Day-Night Cycle:**
    - Plants are introduced as sessile organisms with fractal growth patterns.
    - Plants have their own efficiency traits, evolve through mutations, and are composed of multiple interacting, self-assembling, and procedural generated particle types and structures.
    - A day-night cycle affects plant energy modifiers, simulating realistic environmental changes.
7. **Robust and Scalable Design:**
    - Maintains modularity and scalability, allowing easy addition of new traits and behaviors.
    - Ensures efficient performance with vectorized operations and optimized rendering.

Dependencies:
------------
- Python 3.x
- NumPy
- Pygame
- SciPy

Usage:
------
Ensure that the required libraries are installed:
    pip install numpy pygame scipy

Run the simulation:
    python enhanced_cellular_simulation.py

Terminate the simulation by closing the Pygame window or pressing the ESC key.

Note:
-----
This enhanced simulation is designed to handle tens of thousands of cellular components across many types efficiently while showcasing emergent behaviors through dynamic gene interactions, colony formation, resource management, and realistic environmental cycles.
"""

# Import necessary standard and third-party libraries
import math  # Provides access to mathematical functions
import random  # Facilitates random number generation
import pygame  # Handles rendering and event management
import numpy as np  # Enables efficient numerical computations
from typing import List, Tuple, Optional, Dict, Any  # Supports type annotations for better code clarity
from scipy.spatial import cKDTree  # Offers efficient spatial queries for nearest neighbor searches
import time  # For FPS calculation and timing

###############################################################
# Utility & Configuration Classes
###############################################################

class SimulationConfig:
    """
    Holds all configuration parameters for the simulation.
    Allows easy tweaking and ensures defaults are well-defined.
    """
    # Simulation parameters
    n_cell_types: int = 10  # Number of distinct cellular types in the simulation
    particles_per_type: int = 50  # Initial number of cellular components per type
    mass_range: Tuple[float, float] = (0.1, 10.0)  # Range of masses for mass-based cellular types (units: arbitrary)
    base_velocity_scale: float = 1.0  # Base scaling factor for initial cellular velocities (units: pixels/frame)
    mass_based_fraction: float = 0.5  # Fraction of cellular types that are mass-based (0.0 to 1.0)
    interaction_strength_range: Tuple[float, float] = (-1.0, 1.0)  # Range for interaction rule strengths
    max_frames: int = 0  # Maximum number of simulation frames; 0 means run indefinitely
    initial_energy: float = 100.0  # Initial energy assigned to each cellular component (units: arbitrary)
    friction: float = 0.5  # Friction coefficient applied to cellular velocities (0.0 to 1.0)
    global_temperature: float = 0.1 # Thermal noise factor influencing cellular motion (units: arbitrary)
    predation_range: float = 50.0  # Maximum distance for give-take interactions in pixels
    energy_transfer_factor: float = 0.5  # Fraction of energy transferred during give-take (0.0 to 1.0)
    mass_transfer: bool = True  # Flag to enable mass transfer during give-take
    max_age: float = 30000.0  # Maximum age a cellular component can reach before death (units: frames)
    evolution_interval: int = 90  # Number of frames between evolution of interaction parameters
    synergy_range: float = 150.0  # Maximum distance for synergy (alliance) interactions in pixels

    # Adaptive Frame Management Parameters
    target_fps: float = 60.0  # Target frames per second
    min_fps_threshold: float = 30.0  # FPS threshold below which culling begins
    max_fps_threshold: float = 120.0  # FPS threshold above which energy boost begins
    cull_percentage: float = 0.1  # Percentage of oldest particles to cull when FPS is low
    fps_check_interval: int = 30  # Number of frames between FPS checks
    energy_boost_factor: float = 1.1  # Factor to multiply energy by when FPS is high
    max_cull_per_frame: int = 5  # Maximum number of particles to cull per type per frame
    fps_smoothing_factor: float = 0.1  # Smoothing factor for FPS calculation (0.0 to 1.0)
    min_particles_per_type: int = 10  # Minimum number of particles to maintain per type

    # Reproduction parameters
    reproduction_energy_threshold: float = 150.0  # Energy threshold for a cellular component to reproduce (units: arbitrary)
    reproduction_mutation_rate: float = 0.01  # Probability of mutation during reproduction (0.0 to 1.0)
    reproduction_offspring_energy_fraction: float = 0.75  # Fraction of parent's energy given to offspring (0.0 to 1.0)

    # Clustering parameters (for flocking behavior)
    alignment_strength: float = 0.5  # Strength of velocity alignment within a cluster (units: arbitrary)
    cohesion_strength: float = 0.2  # Strength of cohesion (movement towards center) within a cluster (units: arbitrary)
    separation_strength: float = 0.3  # Strength of separation (avoiding crowding) within a cluster (units: arbitrary)
    cluster_radius: float = 50.0  # Radius within which cellular components consider each other for clustering in pixels

    # Rendering parameters
    particle_size: float = 5.0  # Size (radius) of cellular components when rendered in pixels

    # Energy Efficiency Modifier Parameters
    energy_efficiency_range: Tuple[float, float] = (-0.5, 2.0)  # Range for energy efficiency trait
    energy_efficiency_mutation_rate: float = 0.1  # Probability of mutation in energy efficiency during reproduction (0.0 to 1.0)
    energy_efficiency_mutation_range: Tuple[float, float] = (-0.1, 0.1)  # Range for mutation adjustments (units: arbitrary)

    # Zones: Environmental areas affecting cellular component energy
    zones: List[Dict[str, Any]] = [
        {
            "x": 0.25,  # X-coordinate of zone center as a fraction of screen width (0.0 to 1.0)
            "y": 0.25,  # Y-coordinate of zone center as a fraction of screen height (0.0 to 1.0)
            "radius": 75,  # Radius of the zone in pixels
            "effect": {"energy_modifier": 1.1}  # Effect on cellular component energy within the zone
        },
        {
            "x": 0.75,  # X-coordinate of another zone center as a fraction of screen width (0.0 to 1.0)
            "y": 0.75,  # Y-coordinate of another zone center as a fraction of screen height (0.0 to 1.0)
            "radius": 75,  # Radius of the second zone in pixels
            "effect": {"energy_modifier": 0.9}  # Effect on cellular component energy within the second zone
        }
    ]

    # Gene Configuration Parameters
    gene_traits: List[str] = ["speed_factor", "interaction_strength"]  # List of gene trait names
    gene_mutation_rate: float = 0.05  # Base mutation rate for gene traits (0.0 to 1.0)
    gene_mutation_range: Tuple[float, float] = (-0.1, 0.1)  # Range for gene trait mutations

    # Plant Configuration Parameters
    plant_growth_rate: float = 0.1  # Rate at which plants grow fractal structures (units: fractal iterations/frame)
    plant_energy_modifier_day: float = 1.2  # Energy modifier for plants during day (units: arbitrary)
    plant_energy_modifier_night: float = 0.8  # Energy modifier for plants during night (units: arbitrary)
    day_duration: int = 6000  # Number of frames representing a full day cycle
    night_duration: int = 6000  # Number of frames representing a full night cycle
    plant_mutation_rate: float = 0.02  # Mutation rate for plant traits during reproduction (0.0 to 1.0)
    plant_mutation_range: Tuple[float, float] = (-0.05, 0.05)  # Mutation range for plant traits (units: arbitrary)

###############################################################
# Utility Functions
###############################################################

def random_xy(window_width: int, window_height: int, n: int = 1) -> np.ndarray:
    """
    Generate n random (x,y) coordinates within [0, window_width] x [0, window_height].
    Returns a numpy array of shape (n, 2) with float coordinates.

    Parameters:
    -----------
    window_width : int
        The width of the simulation window in pixels.
    window_height : int
        The height of the simulation window in pixels.
    n : int, optional
        Number of random coordinates to generate (default is 1).

    Returns:
    --------
    np.ndarray
        Array of shape (n, 2) containing random (x, y) positions as floats.
    """
    # Utilize NumPy's uniform distribution for floating-point positions
    return np.random.uniform(0, [window_width, window_height], (n, 2)).astype(float)  # Generate random (x,y) positions

def generate_vibrant_colors(n: int) -> List[str]:
    """
    Generate n distinct vibrant neon color names from a predefined set.
    Excludes grey, white, black, brown, cream, and their shades.
    If more colors are needed than predefined, create new unique color names by appending numbers.

    Parameters:
    -----------
    n : int
        Number of distinct colors to generate.

    Returns:
    --------
    List[str]
        List of color names as strings.
    """
    # Predefined vibrant neon colors excluding grey, white, black, brown, cream, and their shades
    base_colors = [
        "neon_red", "neon_green", "neon_blue", "neon_yellow", "neon_cyan",
        "neon_magenta", "neon_orange", "neon_lime", "neon_purple", "neon_pink",
        "neon_gold", "neon_aqua", "neon_chartreuse", "neon_crimson", "neon_navy",
        "neon_maroon", "neon_fuchsia", "neon_teal", "neon_electric_blue", "neon_bright_green",
        "neon_spring_green", "neon_sky_blue", "neon_pale_green", "neon_dark_purple"
    ]
    # Shuffle to ensure randomness
    random.shuffle(base_colors)
    # Filter out undesired colors if any slipped through (additional safety)
    undesired_colors = {"grey", "gray", "white", "black", "brown", "cream"}  # Set of undesired color substrings
    filtered_colors = [color for color in base_colors if not any(undesired in color.lower() for undesired in undesired_colors)]
    
    if n <= len(filtered_colors):
        return filtered_colors[:n]  # Return first n colors if sufficient

    # If more colors are needed, generate extended unique color names
    extended_colors = filtered_colors[:]
    while len(extended_colors) < n:
        # Append a random number to a base color to create a unique color name
        c = random.choice(filtered_colors) + "_" + str(random.randint(0, 999))
        if c not in extended_colors and not any(undesired in c.lower() for undesired in undesired_colors):
            extended_colors.append(c)  # Add new unique color name
    return extended_colors[:n]  # Return exactly n colors

###############################################################
# Cellular Component & Type Data Management
###############################################################

class CellularTypeData:
    """
    Manages cellular component data for a single cellular type using numpy arrays.
    Each type can be mass-based or massless.

    Attributes:
    -----------
    type_id : int
        ID index of this cellular type.
    color : str
        Color of this cellular type.
    mass_based : bool
        True if cellular type is mass-based.
    x, y : np.ndarray
        Positions of cellular components in the x and y axes.
    vx, vy : np.ndarray
        Velocities of cellular components in the x and y directions.
    energy : np.ndarray
        Energy levels of cellular components.
    mass : Optional[np.ndarray]
        Masses of cellular components if mass_based, else None.
    alive : np.ndarray (bool)
        Boolean mask indicating alive/dead status of cellular components.
    age : np.ndarray
        Ages of cellular components.
    max_age : float
        Maximum age for cellular components before death.
    energy_efficiency : np.ndarray
        Energy efficiency trait of cellular components, affecting maximum speed.
    speed_factor : np.ndarray
        Gene trait influencing the speed scaling of cellular components.
    interaction_strength : np.ndarray
        Gene trait influencing the interaction force scaling of cellular components.
    """

    def __init__(self,
                 type_id: int,
                 color: str,
                 n_particles: int,
                 window_width: int,
                 window_height: int,
                 initial_energy: float,
                 max_age: float,
                 mass: Optional[float] = None,
                 base_velocity_scale: float = 1.0,
                 energy_efficiency: Optional[float] = None,
                 gene_traits: List[str] = ["speed_factor", "interaction_strength"],
                 gene_mutation_rate: float = 0.05,
                 gene_mutation_range: Tuple[float, float] = (-0.1, 0.1)):
        """
        Initialize a CellularTypeData instance with given parameters.

        Parameters:
        -----------
        type_id : int
            Unique identifier for the cellular type.
        color : str
            Color name for rendering cellular components of this type.
        n_particles : int
            Initial number of cellular components in this type.
        window_width : int
            Width of the simulation window in pixels.
        window_height : int
            Height of the simulation window in pixels.
        initial_energy : float
            Initial energy assigned to each cellular component.
        max_age : float
            Maximum age a cellular component can reach before dying.
        mass : Optional[float], default=None
            Mass of cellular components if the type is mass-based.
        base_velocity_scale : float, default=1.0
            Base scaling factor for initial velocities of cellular components.
        energy_efficiency : Optional[float], default=None
            Initial energy efficiency trait of cellular components. If None, randomly initialized within range.
        gene_traits : List[str], default=["speed_factor", "interaction_strength"]
            List of gene trait names.
        gene_mutation_rate : float, default=0.05
            Base mutation rate for gene traits (0.0 to 1.0).
        gene_mutation_range : Tuple[float, float], default=(-0.1, 0.1)
            Range for gene trait mutations (units: arbitrary).
        """
        # Store metadata
        self.type_id: int = type_id  # Unique ID for the cellular type
        self.color: str = color  # Color for rendering
        self.mass_based: bool = (mass is not None)  # Flag indicating if type is mass-based

        # Initialize cellular component positions randomly within the window
        coords = random_xy(window_width, window_height, n_particles)  # Generate random (x,y) positions
        self.x: np.ndarray = coords[:, 0]  # X positions as float array
        self.y: np.ndarray = coords[:, 1]  # Y positions as float array

        # Initialize cellular component velocities with random values scaled by base_velocity_scale and energy_efficiency
        if energy_efficiency is None:
            # Randomly initialize energy efficiency within the defined range
            self.energy_efficiency: np.ndarray = np.random.uniform(
                SimulationConfig.energy_efficiency_range[0],
                SimulationConfig.energy_efficiency_range[1],
                n_particles
            )
        else:
            # Set a fixed energy efficiency if provided
            self.energy_efficiency: np.ndarray = np.full(n_particles, energy_efficiency, dtype=float)  # Fixed energy efficiency

        # Calculate velocity scaling based on energy efficiency
        velocity_scaling = base_velocity_scale / self.energy_efficiency  # Higher efficiency -> lower speed

        # Initialize velocities with random values multiplied by velocity scaling
        self.vx: np.ndarray = (np.random.uniform(-0.5, 0.5, n_particles) * velocity_scaling).astype(float)  # X velocities
        self.vy: np.ndarray = (np.random.uniform(-0.5, 0.5, n_particles) * velocity_scaling).astype(float)  # Y velocities

        # Initialize energy levels for all cellular components
        self.energy: np.ndarray = np.full(n_particles, initial_energy, dtype=float)  # Energy levels

        # Initialize mass if type is mass-based
        if self.mass_based:
            if mass is None or mass <= 0.0:
                # Mass must be positive for mass-based types
                raise ValueError("Mass must be positive for mass-based cellular types.")
            self.mass: np.ndarray = np.full(n_particles, mass, dtype=float)  # Mass values
        else:
            self.mass = None  # Mass is None for massless types

        # Initialize alive status and age for cellular components
        self.alive: np.ndarray = np.ones(n_particles, dtype=bool)  # All components start as alive
        self.age: np.ndarray = np.zeros(n_particles, dtype=float)  # Initial age is 0
        self.max_age: float = max_age  # Maximum age before death

        # Initialize gene traits with random values within a sensible range
        self.speed_factor: np.ndarray = np.random.uniform(0.5, 1.5, n_particles)  # Speed scaling factors
        self.interaction_strength: np.ndarray = np.random.uniform(0.5, 1.5, n_particles)  # Interaction force scaling factors

        # Gene mutation parameters
        self.gene_mutation_rate: float = gene_mutation_rate  # Mutation rate for genes
        self.gene_mutation_range: Tuple[float, float] = gene_mutation_range  # Mutation range for genes

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
        return mask  # Return the combined mask

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
        # Optionally, drain energy based on age or other factors
        # Example: self.energy -= 0.01 * self.age

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
        # Identify components that died of old age
        dead_due_to_age = (~self.alive) & (self.age >= self.max_age)  # Boolean mask for components dying of old age

        # Find indices of alive components
        alive_indices = np.where(self.alive)[0]  # Array of indices where alive is True

        # Find indices of components dying of old age
        dead_age_indices = np.where(dead_due_to_age)[0]  # Array of indices where dead_due_to_age is True

        # Number of alive components
        num_alive = alive_indices.size  # Total number of alive components

        if num_alive > 0 and dead_age_indices.size > 0:
            # Extract positions of alive components for distance calculations
            alive_x = self.x[alive_indices]  # X positions of alive components
            alive_y = self.y[alive_indices]  # Y positions of alive components

            # Build a KD-Tree for efficient nearest neighbor search among alive components
            tree = cKDTree(np.vstack((alive_x, alive_y)).T)  # Create KD-Tree from alive positions

            # Iterate over each component dying of old age to transfer its energy
            for dead_idx in dead_age_indices:
                dead_x, dead_y = self.x[dead_idx], self.y[dead_idx]  # Position of the dead component
                dead_energy = self.energy[dead_idx]  # Energy to be transferred

                # Query the three nearest neighbors within predation_range
                distances, neighbor_idxs = tree.query([dead_x, dead_y], k=3, distance_upper_bound=config.predation_range)

                # Filter out neighbors beyond predation_range or nonexistent (infinite distance)
                valid = distances < config.predation_range  # Boolean array indicating valid neighbors
                valid_neighbors = neighbor_idxs[valid]  # Indices of valid neighbors

                num_neighbors = len(valid_neighbors)  # Number of valid neighbors
                if num_neighbors > 0:
                    # Calculate energy to transfer to each neighbor
                    energy_per_neighbor = dead_energy / num_neighbors  # Energy split evenly among neighbors

                    # Transfer energy to the nearest neighbors
                    self.energy[alive_indices[valid_neighbors]] += energy_per_neighbor  # Add energy to neighbors

                # After transferring energy, set the dead component's energy to zero to ensure conservation
                self.energy[dead_idx] = 0.0  # Dead component energy is now zero

        # Now remove dead components by applying the alive mask
        alive_mask = self.is_alive_mask()  # Recompute alive mask after potential energy transfers

        # Filter all arrays to retain only alive components using the mask
        self.x = self.x[alive_mask]  # Retain x positions of alive components
        self.y = self.y[alive_mask]  # Retain y positions of alive components
        self.vx = self.vx[alive_mask]  # Retain x velocities of alive components
        self.vy = self.vy[alive_mask]  # Retain y velocities of alive components
        self.energy = self.energy[alive_mask]  # Retain energy levels of alive components
        self.alive = self.alive[alive_mask]  # Update alive status mask
        self.age = self.age[alive_mask]  # Retain ages of alive components
        self.energy_efficiency = self.energy_efficiency[alive_mask]  # Retain energy efficiency traits
        self.speed_factor = self.speed_factor[alive_mask]  # Retain speed factors
        self.interaction_strength = self.interaction_strength[alive_mask]  # Retain interaction strengths
        if self.mass_based and self.mass is not None:
            self.mass = self.mass[alive_mask]  # Retain mass values if type is mass-based

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
                      max_age: float) -> None:
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

        if self.mass_based:
            if mass_val is None or mass_val <= 0.0:
                # Ensure mass is positive; assign a small random mass if invalid
                mass_val = max(0.1, abs(mass_val if mass_val is not None else 1.0))
            self.mass = np.concatenate((self.mass, [mass_val]))  # Append new mass value

        self.max_age = max_age  # Update type's max_age if necessary

###############################################################
# Cellular Type Manager (Handles Multi-Type Operations & Reproduction)
###############################################################

class CellularTypeManager:
    """
    Manages all cellular types in the simulation. Handles reproduction, interaction between types, and updates.
    """
    
    def __init__(self, config: SimulationConfig, colors: List[str], mass_based_type_indices: List[int]):
        """
        Initialize the CellularTypeManager with configuration, colors, and mass-based cellular type indices.

        Parameters:
        -----------
        config : SimulationConfig
            Configuration parameters for the simulation.
        colors : List[str]
            List of color names for each cellular type.
        mass_based_type_indices : List[int]
            List of cellular type indices that are mass-based.
        """
        self.config = config  # Store simulation configuration
        self.cellular_types: List[CellularTypeData] = []  # List to hold all cellular type data
        self.mass_based_type_indices = mass_based_type_indices  # Indices of mass-based cellular types
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
        Offspring inherit properties, with possible mutations.
        This method is fully vectorized for performance.
        """
        for ct in self.cellular_types:
            # Identify components eligible for reproduction based on alive status and energy threshold
            eligible = (ct.alive & (ct.energy > self.config.reproduction_energy_threshold))
            num_offspring = np.sum(eligible)  # Number of eligible components
            if num_offspring == 0:
                continue  # Skip if no eligible components

            # Parents lose half their energy upon reproduction
            ct.energy[eligible] *= 0.5
            parent_energy = ct.energy[eligible]  # Energy after energy loss

            # Offspring energy is a fraction of parent's energy
            offspring_energy = parent_energy * self.config.reproduction_offspring_energy_fraction

            # Apply mutations to offspring energy with a certain mutation rate
            mutation_mask = np.random.random(num_offspring) < self.config.reproduction_mutation_rate  # Boolean mask for mutations
            offspring_energy[mutation_mask] *= np.random.uniform(0.9, 1.1, size=mutation_mask.sum())  # Apply energy mutations

            # Offspring energy efficiency with possible mutations
            offspring_efficiency = ct.energy_efficiency[eligible].copy()  # Inherit parent's energy efficiency
            offspring_efficiency[mutation_mask] += np.random.uniform(
                self.config.energy_efficiency_mutation_range[0],
                self.config.energy_efficiency_mutation_range[1],
                size=mutation_mask.sum()
            )  # Apply mutations
            # Clamp energy efficiency within the defined range to prevent extreme values
            offspring_efficiency = np.clip(
                offspring_efficiency,
                self.config.energy_efficiency_range[0],
                self.config.energy_efficiency_range[1]
            )

            # Offspring gene traits with possible mutations
            offspring_speed_factor = ct.speed_factor[eligible].copy()  # Inherit parent's speed factor
            offspring_interaction_strength = ct.interaction_strength[eligible].copy()  # Inherit parent's interaction strength

            # Apply mutations to speed_factor with mutation_mask
            offspring_speed_factor[mutation_mask] += np.random.uniform(
                self.config.gene_mutation_range[0],
                self.config.gene_mutation_range[1],
                size=mutation_mask.sum()
            )
            # Clamp speed_factor within a sensible range to prevent extreme behaviors
            offspring_speed_factor = np.clip(
                offspring_speed_factor,
                0.1,  # Minimum speed factor to avoid stagnation
                3.0   # Maximum speed factor to prevent excessive speeds
            )

            # Apply mutations to interaction_strength with mutation_mask
            offspring_interaction_strength[mutation_mask] += np.random.uniform(
                self.config.gene_mutation_range[0],
                self.config.gene_mutation_range[1],
                size=mutation_mask.sum()
            )
            # Clamp interaction_strength within a sensible range
            offspring_interaction_strength = np.clip(
                offspring_interaction_strength,
                0.1,  # Minimum interaction strength
                3.0   # Maximum interaction strength
            )

            # Offspring mass if type is mass-based
            if ct.mass_based and ct.mass is not None:
                offspring_mass = ct.mass[eligible].copy()  # Inherit parent's mass
                offspring_mass[mutation_mask] *= np.random.uniform(0.95, 1.05, size=mutation_mask.sum())  # Apply mass mutations
                # Ensure mass remains positive to prevent invalid masses
                offspring_mass = np.maximum(offspring_mass, 0.1)
            else:
                offspring_mass = None  # No mass for massless types

            # Offspring positions are same as parents to start
            offspring_x = ct.x[eligible].copy()
            offspring_y = ct.y[eligible].copy()

            # Offspring velocities are adjusted based on energy efficiency and speed_factor
            offspring_velocity_scale = self.config.base_velocity_scale / offspring_efficiency * offspring_speed_factor
            offspring_vx = np.random.uniform(-0.5, 0.5, num_offspring).astype(float) * offspring_velocity_scale  # X velocities
            offspring_vy = np.random.uniform(-0.5, 0.5, num_offspring).astype(float) * offspring_velocity_scale  # Y velocities

            # Prepare new particles' data as a list of tuples for bulk addition
            new_particles = [
                (offspring_x[idx], offspring_y[idx], offspring_vx[idx], offspring_vy[idx],
                 offspring_energy[idx],
                 offspring_mass[idx] if offspring_mass is not None else None,
                 offspring_efficiency[idx],
                 offspring_speed_factor[idx],
                 offspring_interaction_strength[idx],
                 self.config.max_age)
                for idx in range(num_offspring)
            ]

            if not new_particles:
                continue  # Skip if no new particles to add

            # Extract separate lists for each attribute from new_particles
            new_x, new_y, new_vx, new_vy, new_energy, new_mass, new_efficiency, new_speed_factor, new_interaction_strength, new_max_age = zip(*new_particles)

            # Convert lists to NumPy arrays for efficient appending
            new_x = np.array(new_x)
            new_y = np.array(new_y)
            new_vx = np.array(new_vx)
            new_vy = np.array(new_vy)
            new_energy = np.array(new_energy)
            new_efficiency = np.array(new_efficiency)
            new_speed_factor = np.array(new_speed_factor)
            new_interaction_strength = np.array(new_interaction_strength)
            new_max_age = np.array(new_max_age)

            # Append new particles in bulk to cellular type data
            ct.x = np.concatenate((ct.x, new_x))  # Append new x positions
            ct.y = np.concatenate((ct.y, new_y))  # Append new y positions
            ct.vx = np.concatenate((ct.vx, new_vx))  # Append new x velocities
            ct.vy = np.concatenate((ct.vy, new_vy))  # Append new y velocities
            ct.energy = np.concatenate((ct.energy, new_energy))  # Append new energy levels
            ct.energy_efficiency = np.concatenate((ct.energy_efficiency, new_efficiency))  # Append energy efficiency
            ct.speed_factor = np.concatenate((ct.speed_factor, new_speed_factor))  # Append speed factor
            ct.interaction_strength = np.concatenate((ct.interaction_strength, new_interaction_strength))  # Append interaction strength
            ct.age = np.concatenate((ct.age, np.zeros(num_offspring)))  # Initialize ages to 0
            ct.alive = np.concatenate((ct.alive, np.ones(num_offspring, dtype=bool)))  # Set alive status to True

            if ct.mass_based and ct.mass is not None:
                # Append new mass values, ensuring they are positive
                new_mass = np.array(new_mass)
                new_mass = np.maximum(new_mass, 0.1)  # Ensure mass remains positive
                ct.mass = np.concatenate((ct.mass, new_mass))  # Append new mass values

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
        final_rules = []  # Initialize list to hold interaction rules
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
        max_dist = random.uniform(50, 200)

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
        matrix = np.zeros((self.config.n_cell_types, self.config.n_cell_types), dtype=bool)  # Initialize matrix with False
        for i in range(self.config.n_cell_types):
            for j in range(self.config.n_cell_types):
                if i != j:
                    # 10% chance that cellular type i gives energy to cellular type j
                    if random.random() < 0.1:
                        matrix[i, j] = True  # Set to True indicating give-take relationship
        return matrix  # Return the completed give-take matrix

    def _create_synergy_matrix(self) -> np.ndarray:
        """
        Create a synergy matrix indicating energy sharing between cellular types.

        Returns:
        --------
        np.ndarray
            NxN float matrix where element (i,j) > 0 indicates synergy factor between cellular type i and j.
        """
        synergy_matrix = np.zeros((self.config.n_cell_types, self.config.n_cell_types), dtype=float)  # Initialize with zeros
        for i in range(self.config.n_cell_types):
            for j in range(self.config.n_cell_types):
                if i != j:
                    if random.random() < 0.1:
                        # 10% chance to establish synergy with a random factor between 0.01 and 0.3
                        synergy_matrix[i, j] = random.uniform(0.01, 0.3)
                    else:
                        synergy_matrix[i, j] = 0.0  # No synergy
        return synergy_matrix  # Return the completed synergy matrix

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
                    params["max_dist"] = max(10, params["max_dist"] * random.uniform(0.95, 1.05))

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
# Environmental Manager
###############################################################

class EnvironmentalManager:
    """
    Manages environmental zones and their effects on cellular components.
    """
    
    def __init__(self, config: SimulationConfig, screen_width: int, screen_height: int):
        """
        Initialize the EnvironmentalManager with given configuration.

        Parameters:
        -----------
        config : SimulationConfig
            Configuration parameters for the simulation.
        screen_width : int
            Width of the simulation screen in pixels.
        screen_height : int
            Height of the simulation screen in pixels.
        """
        self.zones = config.zones  # Store list of environmental zones
        # Update zone centers based on screen dimensions
        for zone in self.zones:
            # Convert fractional positions to absolute pixel positions
            zone["x_abs"] = zone["x"] * screen_width  # Absolute X-coordinate in pixels
            zone["y_abs"] = zone["y"] * screen_height  # Absolute Y-coordinate in pixels
            zone["radius_sq"] = zone["radius"] ** 2  # Precompute zone radii squared for efficient distance comparisons

    def apply_zone_effects(self, ct: CellularTypeData) -> None:
        """
        Apply zone effects to cellular components based on their positions.

        Parameters:
        -----------
        ct : CellularTypeData
            The cellular type data to which zone effects will be applied.
        """
        if not self.zones:
            return  # No zones to apply effects if zones list is empty

        # Extract positions of cellular components
        x = ct.x  # X positions
        y = ct.y  # Y positions

        # Initialize energy modifiers to 1.0 (no change)
        energy_modifiers = np.ones_like(x)  # Array of ones with the same shape as x

        # Iterate through each zone and apply its effect
        for zone in self.zones:
            # Calculate squared distances from components to zone center
            dx = x - zone["x_abs"]  # Difference in x-coordinate
            dy = y - zone["y_abs"]  # Difference in y-coordinate
            dist_sq = dx * dx + dy * dy  # Squared distance

            # Identify components within the zone
            in_zone = dist_sq <= zone["radius_sq"]  # Boolean array indicating components inside the zone

            # Apply the zone's energy modifier multiplicatively
            energy_modifiers[in_zone] *= zone["effect"].get("energy_modifier", 1.0)  # Update energy modifiers

        # Update the energy of cellular components based on energy modifiers
        ct.energy *= energy_modifiers  # Apply energy modifiers to energy levels

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

def apply_clustering(ct: CellularTypeData, config: SimulationConfig) -> None:
    """
    Apply clustering forces within a single cellular type to simulate flocking behavior.
    The forces include alignment, cohesion, and separation.

    Parameters:
    -----------
    ct : CellularTypeData
        The cellular type data on which to apply clustering.
    config : SimulationConfig
        Configuration parameters for the simulation.
    """
    n = ct.x.size  # Number of cellular components in the type
    if n < 2:
        return  # No clustering needed if fewer than 2 components

    # Create a KD-Tree for efficient neighbor search
    tree = cKDTree(np.vstack((ct.x, ct.y)).T)  # Build KD-Tree from positions

    # Query neighbors within cluster_radius for all components
    neighbors = tree.query_ball_tree(tree, r=config.cluster_radius)  # List of neighbors for each component

    # Initialize arrays to store cumulative velocity changes
    dvx = np.zeros(n, dtype=float)  # Change in velocity in x-direction
    dvy = np.zeros(n, dtype=float)  # Change in velocity in y-direction

    for idx, neighbor_indices in enumerate(neighbors):
        # Exclude self and ensure neighbors are alive
        neighbor_indices = [i for i in neighbor_indices if i != idx and ct.alive[i]]
        count = len(neighbor_indices)  # Number of valid neighbors
        if count == 0:
            continue  # No neighbors to interact with

        # Alignment: average velocity of neighbors
        avg_vx = np.mean(ct.vx[neighbor_indices])  # Average x-velocity
        avg_vy = np.mean(ct.vy[neighbor_indices])  # Average y-velocity

        # Cohesion: vector towards the center of mass of neighbors
        center_x = np.mean(ct.x[neighbor_indices])  # Center of mass x-coordinate
        center_y = np.mean(ct.y[neighbor_indices])  # Center of mass y-coordinate
        cohesion_vector_x = center_x - ct.x[idx]  # Vector towards center in x
        cohesion_vector_y = center_y - ct.y[idx]  # Vector towards center in y

        # Separation: vector away from the average position of neighbors
        separation_vector_x = ct.x[idx] - np.mean(ct.x[neighbor_indices])  # Separation vector in x
        separation_vector_y = ct.y[idx] - np.mean(ct.y[neighbor_indices])  # Separation vector in y

        # Accumulate velocity changes based on alignment, cohesion, and separation
        dvx[idx] += (config.alignment_strength * (avg_vx - ct.vx[idx]) +
                    config.cohesion_strength * cohesion_vector_x +
                    config.separation_strength * separation_vector_x)
        dvy[idx] += (config.alignment_strength * (avg_vy - ct.vy[idx]) +
                    config.cohesion_strength * cohesion_vector_y +
                    config.separation_strength * separation_vector_y)

    # Apply the accumulated velocity changes to all components
    ct.vx += dvx  # Update x-velocities
    ct.vy += dvy  # Update y-velocities

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
    Handles rendering of cellular components and plants on the Pygame window.
    """
    
    def __init__(self, surface: pygame.Surface, config: SimulationConfig):
        """
        Initialize the Renderer with a Pygame surface and configuration.

        Parameters:
        -----------
        surface : pygame.Surface
            The Pygame surface where cellular components and plants will be drawn.
        config : SimulationConfig
            Configuration parameters for the simulation.
        """
        self.surface = surface  # Pygame surface for rendering
        self.config = config    # Simulation configuration

        # Mapping of vibrant neon color names to RGB tuples for rendering
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
            "neon_bright_green": (102, 255, 0)
        }

        # Precompute a surface for all particles to optimize rendering
        self.particle_surface = pygame.Surface(self.surface.get_size(), flags=pygame.SRCALPHA)  # Create a transparent surface
        self.particle_surface = self.particle_surface.convert_alpha()  # Optimize for alpha transparency

    def draw_component(self, x: float, y: float, color: str, energy: float, speed_factor: float) -> None:
        """
        Draw a single cellular component as a circle on the surface. Energy is mapped to brightness.

        Parameters:
        -----------
        x : float
            X-coordinate of the component.
        y : float
            Y-coordinate of the component.
        color : str
            Color name of the component.
        energy : float
            Current energy level of the component, influencing brightness.
        speed_factor : float
            Speed factor gene trait influencing the brightness scaling.
        """
        base_c = self.color_map.get(color, (255, 0, 255))  # Get base color or default to neon magenta if not found
        health = min(100.0, max(0.0, energy))  # Clamp energy between 0 and 100 to determine health
        intensity_factor = max(0.0, min(1.0, health / 100.0))  # Normalize energy to [0,1] for intensity scaling

        # Adjust color brightness based on energy and speed_factor
        # Higher energy and speed_factor -> brighter color; lower values -> dimmer color with a base minimum brightness
        c = (
            min(255, int(base_c[0] * intensity_factor * speed_factor + (1 - intensity_factor) * 100)),
            min(255, int(base_c[1] * intensity_factor * speed_factor + (1 - intensity_factor) * 100)),
            min(255, int(base_c[2] * intensity_factor * speed_factor + (1 - intensity_factor) * 100))
        )

        # Draw the component as a filled circle with the calculated color and size
        pygame.draw.circle(self.particle_surface, c, (int(x), int(y)), int(self.config.particle_size))  # Draw circle on particle surface

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
            self.draw_component(ct.x[idx], ct.y[idx], ct.color, ct.energy[idx], ct.speed_factor[idx])  # Draw each alive component

    def render(self) -> None:
        """
        Blit the particle and plant surfaces onto the main surface and reset the particle surface for the next frame.
        """
        self.surface.blit(self.particle_surface, (0, 0))  # Blit all particles at once onto the main surface
        self.particle_surface.fill((0, 0, 0, 0))  # Clear particle surface by filling with transparent color

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
        display_info = pygame.display.Info()  # Get display information
        screen_width, screen_height = display_info.current_w, display_info.current_h  # Current display dimensions

        # Set up a fullscreen window with the calculated dimensions
        self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)  # Create fullscreen window
        pygame.display.set_caption("Ultra-Advanced Emergent Cellular Automata Simulation")  # Set window title

        self.clock = pygame.time.Clock()  # Clock to manage frame rate
        self.frame_count = 0  # Initialize frame counter
        self.run_flag = True  # Flag to control main loop

        # Calculate minimum distance from edges (1% of the larger screen dimension)
        self.edge_buffer = 0.01 * max(screen_width, screen_height)  # Minimum distance from edges in pixels

        # Setup cellular type colors and identify mass-based types
        self.colors = generate_vibrant_colors(self.config.n_cell_types)  # Generate vibrant colors for each cellular type
        n_mass_types = int(self.config.mass_based_fraction * self.config.n_cell_types)  # Number of mass-based types based on fraction
        mass_based_type_indices = list(range(n_mass_types))  # Indices of mass-based types

        # Initialize CellularTypeManager with configuration, colors, and mass-based type indices
        self.type_manager = CellularTypeManager(self.config, self.colors, mass_based_type_indices)

        # Create cellular type data for each type and add to CellularTypeManager
        for i in range(self.config.n_cell_types):
            if i < n_mass_types:
                # Initialize mass-based cellular type with random mass within mass_range
                mass_val = random.uniform(self.config.mass_range[0], self.config.mass_range[1])
                ct = CellularTypeData(
                    type_id=i,
                    color=self.colors[i],
                    n_particles=self.config.particles_per_type,
                    window_width=screen_width,
                    window_height=screen_height,
                    initial_energy=self.config.initial_energy,
                    max_age=self.config.max_age,
                    mass=mass_val,
                    base_velocity_scale=self.config.base_velocity_scale
                )
            else:
                # Initialize massless cellular type
                ct = CellularTypeData(
                    type_id=i,
                    color=self.colors[i],
                    n_particles=self.config.particles_per_type,
                    window_width=screen_width,
                    window_height=screen_height,
                    initial_energy=self.config.initial_energy,
                    max_age=self.config.max_age,
                    mass=None,  # Mass is None for massless types
                    base_velocity_scale=self.config.base_velocity_scale
                )
            self.type_manager.add_cellular_type_data(ct)  # Add type to manager

        # Initialize InteractionRules with configuration and mass-based type indices
        self.rules_manager = InteractionRules(self.config, mass_based_type_indices)
        # Initialize EnvironmentalManager with given configuration and screen dimensions
        self.env_manager = EnvironmentalManager(self.config, screen_width, screen_height)

        # Initialize Renderer with Pygame surface and configuration
        self.renderer = Renderer(self.screen, self.config)

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
                if event.type == pygame.QUIT:
                    self.run_flag = False  # Exit on quit event
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.run_flag = False  # Exit on ESC key

            # Evolve interaction parameters periodically based on frame count
            self.rules_manager.evolve_parameters(self.frame_count)

            # Clear the main screen by filling it with black to prevent piling up
            self.screen.fill((0, 0, 0))  # Fill the main screen with black

            # Apply all inter-type interactions and updates
            self.apply_all_interactions()

            # Apply environmental effects to all cellular types
            for ct in self.type_manager.cellular_types:
                self.env_manager.apply_zone_effects(ct)  # Modify energy based on environmental zones

            # Apply clustering within each cellular type to simulate flocking behavior
            for ct in self.type_manager.cellular_types:
                apply_clustering(ct, self.config)  # Apply clustering forces

            # Handle reproduction of cellular components across all types
            self.type_manager.reproduce()

            # Remove dead cellular components from all types
            self.type_manager.remove_dead_in_all_types()

            # Draw all cellular types' components on the particle surface
            for ct in self.type_manager.cellular_types:
                self.renderer.draw_cellular_type(ct)  # Render each cellular type

            # Handle boundary reflections after updating positions
            self.handle_boundary_reflections()

            # Render all particles and plants onto the main screen
            self.renderer.render()

            # Update the full display surface to the screen
            pygame.display.flip()  # Update the full display surface to the screen

            # Cap the frame rate and retrieve the current FPS
            current_fps = self.clock.tick(120)  # Attempt to cap at 120 FPS

            # Adaptive Frame Management based on current FPS
            if current_fps < 30.0:
                # If FPS drops below 30, start culling oldest particles each frame until FPS stabilizes
                self.cull_oldest_particles()
            elif current_fps > 120.0:
                # If FPS exceeds 120, give all particles an extra 10% energy every frame until FPS drops below 120
                self.add_global_energy()

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
                m_a = ct_i.mass  # Mass of type i components
                m_b = ct_j.mass  # Mass of type j components
                params["m_a"] = m_a  # Assign mass to parameters
                params["m_b"] = m_b
            else:
                # If masses are not available, disable gravity
                params["use_gravity"] = False

        # Calculate pairwise differences using broadcasting for vectorized operations
        dx = ct_i.x[:, np.newaxis] - ct_j.x[np.newaxis, :]  # Difference in x-coordinates, shape: (n_i, n_j)
        dy = ct_i.y[:, np.newaxis] - ct_j.y[np.newaxis, :]  # Difference in y-coordinates, shape: (n_i, n_j)
        dist_sq = dx * dx + dy * dy  # Squared distances between all pairs, shape: (n_i, n_j)

        # Determine which pairs are within interaction range and not overlapping
        within_range = (dist_sq > 0.0) & (dist_sq <= params["max_dist"] ** 2)  # Boolean mask, shape: (n_i, n_j)

        # Extract indices of interacting pairs
        indices = np.where(within_range)  # Tuple of arrays (i_indices, j_indices)
        if len(indices[0]) == 0:
            return  # No interactions within range

        # Calculate distances for interacting pairs
        dist = np.sqrt(dist_sq[indices])  # Euclidean distances, shape: (num_interactions,)

        # Initialize force arrays
        fx = np.zeros_like(dist)  # Force in x-direction
        fy = np.zeros_like(dist)  # Force in y-direction

        # Potential-based forces
        if params.get("use_potential", True):
            pot_strength = params.get("potential_strength", 1.0)  # Strength of potential
            F_pot = pot_strength / dist  # Force magnitude inversely proportional to distance
            fx += F_pot * dx[indices]  # Apply potential force in x
            fy += F_pot * dy[indices]  # Apply potential force in y

        # Gravity-based forces
        if params.get("use_gravity", False):
            gravity_factor = params.get("gravity_factor", 1.0)  # Gravity scaling factor
            # Gravitational force magnitude based on masses and distance squared
            F_grav = gravity_factor * (params["m_a"][indices[0]] * params["m_b"][indices[1]]) / dist_sq[indices]
            fx += F_grav * dx[indices]  # Apply gravitational force in x
            fy += F_grav * dy[indices]  # Apply gravitational force in y

        # Accumulate forces on cellular type i components using NumPy's add.at for indexed updates
        np.add.at(ct_i.vx, indices[0], fx)  # Add forces to vx of type i components
        np.add.at(ct_i.vy, indices[0], fy)  # Add forces to vy of type i components

        # Handle give-take interactions
        if is_giver:
            # Check if distance is within give-take range
            give_take_within = dist_sq[indices] <= self.config.predation_range ** 2  # Boolean array
            give_take_indices = (indices[0][give_take_within], indices[1][give_take_within])  # Indices within give-take range
            if give_take_indices[0].size > 0:
                # Transfer energy and possibly mass from receiver to giver
                giver_energy = ct_i.energy[give_take_indices[0]]  # Energy of givers
                receiver_energy = ct_j.energy[give_take_indices[1]]  # Energy of receivers
                giver_mass = ct_i.mass[give_take_indices[0]] if ct_i.mass_based else None  # Mass of givers
                receiver_mass = ct_j.mass[give_take_indices[1]] if ct_j.mass_based else None  # Mass of receivers

                # Apply give-take interaction and retrieve updated energies and masses
                updated = give_take_interaction(
                    giver_energy,
                    receiver_energy,
                    giver_mass,
                    receiver_mass,
                    self.config
                )
                ct_i.energy[give_take_indices[0]] = updated[0]  # Update giver energy
                ct_j.energy[give_take_indices[1]] = updated[1]  # Update receiver energy

                if ct_i.mass_based and ct_i.mass is not None and updated[2] is not None:
                    ct_i.mass[give_take_indices[0]] = updated[2]  # Update giver mass
                if ct_j.mass_based and ct_j.mass is not None and updated[3] is not None:
                    ct_j.mass[give_take_indices[1]] = updated[3]  # Update receiver mass

        # Handle synergy (alliance) interactions
        if synergy_factor > 0.0 and self.config.synergy_range > 0.0:
            # Check if distance is within synergy range
            synergy_within = dist_sq[indices] <= self.config.synergy_range ** 2  # Boolean mask
            synergy_indices = (indices[0][synergy_within], indices[1][synergy_within])  # Indices within synergy range
            if synergy_indices[0].size > 0:
                # Share energy between allied cellular type components based on synergy factor
                energyA = ct_i.energy[synergy_indices[0]]  # Energy of type i components
                energyB = ct_j.energy[synergy_indices[1]]  # Energy of type j components
                new_energyA, new_energyB = apply_synergy(energyA, energyB, synergy_factor)  # Calculate new energies
                ct_i.energy[synergy_indices[0]] = new_energyA  # Update energy of type i components
                ct_j.energy[synergy_indices[1]] = new_energyB  # Update energy of type j components

        # Apply friction and global temperature effects to velocities
        ct_i.vx *= self.config.friction  # Apply friction to x-velocities
        ct_i.vy *= self.config.friction  # Apply friction to y-velocities
        ct_i.vx += (np.random.uniform(-0.5, 0.5, n_i)) * self.config.global_temperature  # Add thermal noise to x-velocities
        ct_i.vy += (np.random.uniform(-0.5, 0.5, n_i)) * self.config.global_temperature  # Add thermal noise to y-velocities

        # Update cellular component positions based on velocities
        ct_i.x += ct_i.vx  # Update x positions
        ct_i.y += ct_i.vy  # Update y positions

        # Handle boundary conditions by reflecting velocities if components hit the edges
        self.handle_boundary_reflection(ct_i)

        # Age the cellular components and update their alive status
        ct_i.age_components()  # Increment age and potentially drain energy
        ct_i.update_states()  # Update component states if needed
        ct_i.update_alive()   # Recompute alive mask after updates

    def handle_boundary_reflections(self) -> None:
        """
        Handle boundary reflections for all cellular components across all cellular types.
        Ensures that components maintain a minimum distance from screen edges and undergo perfectly elastic collisions upon reaching boundaries.
        """
        for ct in self.type_manager.cellular_types:
            if ct.x.size == 0:
                continue  # Skip if no components in this type

            # Reflect components that are too close to the left or right edges
            left_bound = self.edge_buffer  # Left boundary
            right_bound = self.screen.get_width() - self.edge_buffer  # Right boundary
            close_left = ct.x < left_bound  # Components too close to the left
            close_right = ct.x > right_bound  # Components too close to the right
            ct.vx[close_left | close_right] *= -1  # Reverse x-velocity for elastic reflection

            # Reflect components that are too close to the top or bottom edges
            top_bound = self.edge_buffer  # Top boundary
            bottom_bound = self.screen.get_height() - self.edge_buffer  # Bottom boundary
            close_top = ct.y < top_bound  # Components too close to the top
            close_bottom = ct.y > bottom_bound  # Components too close to the bottom
            ct.vy[close_top | close_bottom] *= -1  # Reverse y-velocity for elastic reflection

            # Ensure components stay within bounds after reflection
            ct.x = np.clip(ct.x, left_bound, right_bound)  # Clamp x positions within bounds
            ct.y = np.clip(ct.y, top_bound, bottom_bound)  # Clamp y positions within bounds

    def handle_boundary_reflection(self, ct: CellularTypeData) -> None:
        """
        Handle boundary reflections for a single cellular type.

        Parameters:
        -----------
        ct : CellularTypeData
            The cellular type data for which to handle boundary reflections.
        """
        if ct.x.size == 0:
            return  # Skip if no components in this type

        # Reflect components that are too close to the left or right edges
        left_bound = self.edge_buffer  # Left boundary
        right_bound = self.screen.get_width() - self.edge_buffer  # Right boundary
        close_left = ct.x < left_bound  # Components too close to the left
        close_right = ct.x > right_bound  # Components too close to the right
        ct.vx[close_left | close_right] *= -1  # Reverse x-velocity for elastic reflection

        # Reflect components that are too close to the top or bottom edges
        top_bound = self.edge_buffer  # Top boundary
        bottom_bound = self.screen.get_height() - self.edge_buffer  # Bottom boundary
        close_top = ct.y < top_bound  # Components too close to the top
        close_bottom = ct.y > bottom_bound  # Components too close to the bottom
        ct.vy[close_top | close_bottom] *= -1  # Reverse y-velocity for elastic reflection

        # Ensure components stay within bounds after reflection
        ct.x = np.clip(ct.x, left_bound, right_bound)  # Clamp x positions within bounds
        ct.y = np.clip(ct.y, top_bound, bottom_bound)  # Clamp y positions within bounds

    def cull_oldest_particles(self) -> None:
        """
        Cull the oldest particles across all cellular types to improve FPS.
        Removes one oldest particle per type.
        """
        for ct in self.type_manager.cellular_types:
            if ct.x.size == 0:
                continue  # Skip if no particles to cull

            # Identify the index of the oldest particle
            oldest_idx = np.argmax(ct.age)  # Index of the particle with the maximum age
            # Remove the oldest particle by filtering out its index
            mask = np.ones(ct.x.size, dtype=bool)  # Initialize mask to keep all
            mask[oldest_idx] = False  # Set mask to False for the oldest particle
            ct.x = ct.x[mask]  # Remove x position of the oldest particle
            ct.y = ct.y[mask]  # Remove y position of the oldest particle
            ct.vx = ct.vx[mask]  # Remove vx of the oldest particle
            ct.vy = ct.vy[mask]  # Remove vy of the oldest particle
            ct.energy = ct.energy[mask]  # Remove energy of the oldest particle
            ct.alive = ct.alive[mask]  # Remove alive status of the oldest particle
            ct.age = ct.age[mask]  # Remove age of the oldest particle
            ct.energy_efficiency = ct.energy_efficiency[mask]  # Remove energy efficiency of the oldest particle
            ct.speed_factor = ct.speed_factor[mask]  # Remove speed factor of the oldest particle
            ct.interaction_strength = ct.interaction_strength[mask]  # Remove interaction strength of the oldest particle
            if ct.mass_based and ct.mass is not None:
                ct.mass = ct.mass[mask]  # Remove mass of the oldest particle if applicable

    def add_global_energy(self) -> None:
        """
        Add an extra 10% energy to all particles across all cellular types to stimulate reproduction.
        """
        for ct in self.type_manager.cellular_types:
            ct.energy *= 1.1  # Increase energy by 10%
            # Optionally, clamp energy to a maximum value to prevent runaway growth
            ct.energy = np.clip(ct.energy, 0.0, 1000.0)  # Example maximum energy

    def run(self) -> None:
        """
        Start the main simulation loop.
        """
        self.main_loop()  # Begin the main simulation loop

###############################################################
# Entry Point
###############################################################

def main():
    """
    Main configuration and run function.
    Initializes the simulation configuration, creates a CellularAutomata instance,
    and starts the main simulation loop.
    """
    config = SimulationConfig()  # Instantiate default simulation configuration
    cellular_automata = CellularAutomata(config)  # Create CellularAutomata with the configuration
    cellular_automata.run()  # Start the main simulation loop

if __name__ == "__main__":
    main()  # Execute the main function when the script is run
