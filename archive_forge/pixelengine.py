"""
Ultra-Advanced Optimized Cellular Automata and Multicellular Simulation with Pixelengine Rendering
---------------------------------------------------------------------------------------------------
This simulation merges the optimized rendering approach from Pixelengine with an advanced cellular automata system.
It offers unparalleled performance, scalability, and configurability, adhering to the highest coding standards.

**Key Features:**
1. Optimized Bulk Rendering using NumPy and Pygame.
2. Fully Parameterized and Configurable Simulation.
3. Advanced Cellular Interactions and Reproduction Mechanics.
4. Environmental Management with Zones and Day-Night Cycles.
5. Comprehensive Documentation and Modular Design.

**Dependencies:**
- Python 3.x
- NumPy
- Pygame
- SciPy
- scikit-learn

**Usage:**
---------
Ensure required libraries are installed:
    pip install numpy pygame scipy scikit-learn

Run the simulation:
    python optimized_cellular_automata.py

Terminate the simulation by closing the Pygame window or pressing the ESC key.
"""

# Import standard and third-party libraries
import math
import random
import sys
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
        default=10.0,
        metadata={
            "description": "Buffer distance from window edges.",
            "impact": "Prevents components from being rendered on the edges of the window.",
            "units": "Pixels",
            "min": 0.0,
            "max": 100.0,
            "typical": 10.0,
            "default": 10.0
        }
    )

    n_cell_types: int = field(
        default=5,
        metadata={
            "description": "Number of distinct cellular types in the simulation.",
            "impact": "Determines diversity and interactions among different cellular types.",
            "units": "Count (integer)",
            "min": 1,
            "max": 100,
            "typical": 5,
            "default": 5
        }
    )
    particles_per_type: int = field(
        default=100,
        metadata={
            "description": "Initial number of particles per cellular type.",
            "impact": "Sets the initial population size for each cellular type.",
            "units": "Count (integer)",
            "min": 1,
            "max": 10000,
            "typical": 1000,
            "default": 1000
        }
    )
    mass_range: Tuple[float, float] = field(
        default=(0.1, 10.0),
        metadata={
            "description": "Range of masses for mass-based cellular types.",
            "impact": "Defines the possible mass values for cellular types that consider mass.",
            "units": "Mass units (arbitrary)",
            "min": 0.1,
            "max": 100.0,
            "typical": (0.1, 10.0),
            "default": (0.1, 10.0)
        }
    )
    base_velocity_scale: float = field(
        default=1.0,
        metadata={
            "description": "Base scaling factor for initial velocities of cellular components.",
            "impact": "Influences the speed of cellular components upon initialization.",
            "units": "Scalar (dimensionless)",
            "min": 0.1,
            "max": 10.0,
            "typical": 1.0,
            "default": 1.0
        }
    )
    mass_based_fraction: float = field(
        default=0.5,
        metadata={
            "description": "Fraction of cellular types that are mass-based.",
            "impact": "Determines how many cellular types consider mass in their behavior.",
            "units": "Fraction (0.0 to 1.0)",
            "min": 0.0,
            "max": 1.0,
            "typical": 0.5,
            "default": 0.5
        }
    )
    interaction_strength_range: Tuple[float, float] = field(
        default=(-1.0, 1.0),
        metadata={
            "description": "Range for interaction strength between cellular types.",
            "impact": "Defines the potential strength and direction of interactions (attraction/repulsion).",
            "units": "Force units (arbitrary)",
            "min": -10.0,
            "max": 10.0,
            "typical": (-1.0, 1.0),
            "default": (-1.0, 1.0)
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
        default=100.0,
        metadata={
            "description": "Initial energy level assigned to each cellular component.",
            "impact": "Sets the starting energy for cellular components, influencing their survivability and reproduction.",
            "units": "Energy units (arbitrary)",
            "min": 0.0,
            "max": 1000.0,
            "typical": 100.0,
            "default": 100.0
        }
    )
    friction: float = field(
        default=0.25,
        metadata={
            "description": "Friction coefficient applied to cellular velocities each frame.",
            "impact": "Reduces velocities over time, simulating resistance.",
            "units": "Scalar (dimensionless)",
            "min": 0.0,
            "max": 1.0,
            "typical": 0.5,
            "default": 0.5
        }
    )
    global_temperature: float = field(
        default=0.5,  # Prioritized to Version 2's default
        metadata={
            "description": "Global temperature affecting random velocity fluctuations.",
            "impact": "Introduces randomness in cellular movements.",
            "units": "Scalar (dimensionless)",
            "min": 0.0,
            "max": 1.0,
            "typical": 0.05,
            "default": 0.5  # Updated to match the field default
        }
    )
    predation_range: float = field(
        default=50.0,
        metadata={
            "description": "Range within which predation (energy transfer) can occur.",
            "impact": "Determines the proximity required for energy transfer between components.",
            "units": "Pixels",
            "min": 10.0,
            "max": 500.0,
            "typical": 50.0,
            "default": 50.0
        }
    )
    energy_transfer_factor: float = field(
        default=0.5,
        metadata={
            "description": "Fraction of energy transferred during interactions.",
            "impact": "Controls how much energy is moved between cellular components during interactions.",
            "units": "Fraction (0.0 to 1.0)",
            "min": 0.0,
            "max": 1.0,
            "typical": 0.1,
            "default": 0.1
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
        default=30000.0,
        metadata={
            "description": "Maximum lifespan of cellular components in frames.",
            "impact": "Determines how long cellular components can exist before dying of old age.",
            "units": "Frames",
            "min": 1000.0,
            "max": 100000.0,
            "typical": 30000.0,
            "default": 30000.0
        }
    )
    synergy_range: float = field(
        default=150.0,
        metadata={
            "description": "Range within which energy synergy occurs between allied components.",
            "impact": "Defines the proximity required for energy sharing among allied components.",
            "units": "Pixels",
            "min": 10.0,
            "max": 500.0,
            "typical": 150.0,
            "default": 150.0
        }
    )

    # ======================
    # Feature Toggles
    # ======================

    enable_max_age: bool = field(
        default=False,  # Prioritized to Version 2's default
        metadata={
            "description": "Enable or disable maximum age for cellular components.",
            "impact": "Toggles whether cellular components have a lifespan.",
            "units": "Boolean",
            "default": False  # Updated to match the field default
        }
    )
    enable_plants: bool = field(
        default=False,  # Prioritized to Version 2's default
        metadata={
            "description": "Enable or disable plant entities in the simulation.",
            "impact": "Toggles the presence and behavior of plants producing food particles.",
            "units": "Boolean",
            "default": False  # Updated to match the field default
        }
    )
    enable_day_night_cycle: bool = field(
        default=False,  # Prioritized to Version 2's default
        metadata={
            "description": "Enable or disable the day-night cycle.",
            "impact": "Toggles the progression between day and night, affecting plant energy modifiers.",
            "units": "Boolean",
            "default": False  # Updated to match the field default
        }
    )
    enable_food_particles: bool = field(
        default=False,  # Prioritized to Version 2's default
        metadata={
            "description": "Enable or disable food particles in the simulation.",
            "impact": "Toggles the presence of food particles that can be consumed by cellular components.",
            "units": "Boolean",
            "default": False  # Updated to match the field default
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
        default=150.0,
        metadata={
            "description": "Energy threshold required for cellular components to reproduce.",
            "impact": "Determines the minimum energy a component must have to be eligible for reproduction.",
            "units": "Energy units (arbitrary)",
            "min": 50.0,
            "max": 500.0,
            "typical": 150.0,
            "default": 150.0
        }
    )
    reproduction_mutation_rate: float = field(
        default=0.15,  # Prioritized to Version 2's default
        metadata={
            "description": "Probability of mutation occurring during reproduction.",
            "impact": "Controls the likelihood that offspring will undergo genetic mutations.",
            "units": "Fraction (0.0 to 1.0)",
            "min": 0.0,
            "max": 1.0,
            "typical": 0.05,
            "default": 0.15  # Updated to match the field default
        }
    )
    reproduction_offspring_energy_fraction: float = field(
        default=0.25,  # Prioritized to Version 2's default
        metadata={
            "description": "Fraction of parent energy allocated to offspring.",
            "impact": "Determines how much energy is passed from parent to offspring during reproduction.",
            "units": "Fraction (0.0 to 1.0)",
            "min": 0.1,
            "max": 1.0,
            "typical": 0.75,
            "default": 0.25  # Updated to match the field default
        }
    )

    # ======================
    # Clustering Behavior
    # ======================

    alignment_strength: float = field(
        default=0.1,
        metadata={
            "description": "Strength of velocity alignment in clustering behavior.",
            "impact": "Controls how strongly components align their velocities with neighbors.",
            "units": "Scalar (dimensionless)",
            "min": 0.0,
            "max": 1.0,
            "typical": 0.1,
            "default": 0.1
        }
    )
    cohesion_strength: float = field(
        default=0.05,
        metadata={
            "description": "Strength of cohesion in clustering behavior.",
            "impact": "Determines how strongly components move towards the center of their neighbors.",
            "units": "Scalar (dimensionless)",
            "min": 0.0,
            "max": 1.0,
            "typical": 0.05,
            "default": 0.05
        }
    )
    separation_strength: float = field(
        default=0.25,  # Prioritized to Version 2's default
        metadata={
            "description": "Strength of separation in clustering behavior.",
            "impact": "Controls how strongly components avoid overcrowding by separating from neighbors.",
            "units": "Scalar (dimensionless)",
            "min": 0.0,
            "max": 1.0,
            "typical": 0.1,  # Updated to match Version 2's typical
            "default": 0.25  # Updated to match the field default
        }
    )
    cluster_radius: float = field(
        default=50.0,
        metadata={
            "description": "Radius defining neighborhood for clustering behavior.",
            "impact": "Sets the proximity within which components consider others as neighbors for clustering.",
            "units": "Pixels",
            "min": 10.0,
            "max": 200.0,
            "typical": 50.0,
            "default": 50.0
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
            "min": 1.0,
            "max": 10.0,
            "typical": 3.0,
            "default": 3.0
        }
    )
    font_size: int = field(
        default=20,
        metadata={
            "description": "Font size for on-screen text elements like FPS display.",
            "impact": "Controls the readability of textual information rendered on the screen.",
            "units": "Points",
            "min": 10,
            "max": 72,
            "typical": 20,
            "default": 20
        }
    )

    # ======================
    # Energy Efficiency Traits
    # ======================

    energy_efficiency_range: Tuple[float, float] = field(
        default=(0.5, 2.0),
        metadata={
            "description": "Range of energy efficiency traits for cellular components.",
            "impact": "Influences how efficiently components utilize energy, affecting speed and interaction strength.",
            "units": "Scalar (dimensionless)",
            "min": 0.1,
            "max": 5.0,
            "typical": (0.5, 2.0),
            "default": (0.5, 2.0)
        }
    )
    energy_efficiency_mutation_rate: float = field(
        default=0.1,
        metadata={
            "description": "Mutation probability for energy efficiency traits during reproduction.",
            "impact": "Controls the likelihood that offspring will have altered energy efficiency traits.",
            "units": "Fraction (0.0 to 1.0)",
            "min": 0.0,
            "max": 1.0,
            "typical": 0.1,
            "default": 0.1
        }
    )
    energy_efficiency_mutation_range: Tuple[float, float] = field(
        default=(-0.1, 0.1),
        metadata={
            "description": "Range of mutation adjustments for energy efficiency traits.",
            "impact": "Determines how much energy efficiency traits can vary during mutation.",
            "units": "Scalar (dimensionless)",
            "min": -1.0,
            "max": 1.0,
            "typical": (-0.1, 0.1),
            "default": (-0.1, 0.1)
        }
    )

    # ======================
    # Environmental Zones
    # ======================

    zones: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "x": 0.25,  # Relative x position (0.0 to 1.0)
                "y": 0.25,  # Relative y position (0.0 to 1.0)
                "radius": 75.0,  # Radius in pixels
                "effect": {"energy_modifier": 1.01}  # Effect applied within the zone
            },
            {
                "x": 0.75,
                "y": 0.75,
                "radius": 75.0,
                "effect": {"energy_modifier": 0.99}
            }
        ],
        metadata={
            "description": "List of environmental zones with positions, radii, and effects.",
            "impact": "Defines regions in the simulation where cellular components experience modified energy levels.",
            "units": "Relative positions (0.0 to 1.0) for x and y; pixels for radius.",
            "default": [
                {
                    "x": 0.25,
                    "y": 0.25,
                    "radius": 75.0,
                    "effect": {"energy_modifier": 1.01}
                },
                {
                    "x": 0.75,
                    "y": 0.75,
                    "radius": 75.0,
                    "effect": {"energy_modifier": 0.99}
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
        default=0.05,
        metadata={
            "description": "Mutation probability for gene traits during reproduction.",
            "impact": "Controls the likelihood that offspring will undergo mutations in their gene traits.",
            "units": "Fraction (0.0 to 1.0)",
            "min": 0.0,
            "max": 1.0,
            "typical": 0.05,
            "default": 0.05
        }
    )
    gene_mutation_range: Tuple[float, float] = field(
        default=(-0.1, 0.1),
        metadata={
            "description": "Range of mutation adjustments for gene traits.",
            "impact": "Determines how much gene traits can vary during mutation.",
            "units": "Scalar (dimensionless)",
            "min": -1.0,
            "max": 1.0,
            "typical": (-0.1, 0.1),
            "default": (-0.1, 0.1)
        }
    )

    # ======================
    # Plant Configuration
    # ======================

    plant_growth_rate: float = field(
        default=0.1,
        metadata={
            "description": "Rate at which plants grow and produce food particles.",
            "impact": "Controls the frequency and speed of plant growth cycles.",
            "units": "Scalar (frames per growth increment)",
            "min": 0.01,
            "max": 1.0,
            "typical": 0.1,
            "default": 0.1
        }
    )
    plant_energy_modifier_day: float = field(
        default=1.2,
        metadata={
            "description": "Energy modifier applied to plants during the day.",
            "impact": "Enhances plant energy efficiency during daytime.",
            "units": "Scalar (dimensionless)",
            "min": 0.5,
            "max": 2.0,
            "typical": 1.2,
            "default": 1.2
        }
    )
    plant_energy_modifier_night: float = field(
        default=0.8,
        metadata={
            "description": "Energy modifier applied to plants during the night.",
            "impact": "Reduces plant energy efficiency during nighttime.",
            "units": "Scalar (dimensionless)",
            "min": 0.5,
            "max": 2.0,
            "typical": 0.8,
            "default": 0.8
        }
    )
    day_duration: int = field(
        default=6000,
        metadata={
            "description": "Duration of the day phase in frames.",
            "impact": "Sets how long the day lasts before transitioning to night.",
            "units": "Frames",
            "min": 1000,
            "max": 100000,
            "typical": 6000,
            "default": 6000
        }
    )
    night_duration: int = field(
        default=6000,
        metadata={
            "description": "Duration of the night phase in frames.",
            "impact": "Sets how long the night lasts before transitioning to day.",
            "units": "Frames",
            "min": 1000,
            "max": 100000,
            "typical": 6000,
            "default": 6000
        }
    )
    plant_mutation_rate: float = field(
        default=0.02,
        metadata={
            "description": "Mutation probability for plant traits during reproduction.",
            "impact": "Controls the likelihood that plant offspring will undergo genetic mutations.",
            "units": "Fraction (0.0 to 1.0)",
            "min": 0.0,
            "max": 1.0,
            "typical": 0.02,
            "default": 0.02
        }
    )
    plant_mutation_range: Tuple[float, float] = field(
        default=(-0.05, 0.05),
        metadata={
            "description": "Range of mutation adjustments for plant energy efficiency traits.",
            "impact": "Determines how much plant energy efficiency traits can vary during mutation.",
            "units": "Scalar (dimensionless)",
            "min": -1.0,
            "max": 1.0,
            "typical": (-0.05, 0.05),
            "default": (-0.05, 0.05)
        }
    )
    max_growth_stage: int = field(
        default=5,
        metadata={
            "description": "Maximum growth stage a plant can reach.",
            "impact": "Limits the complexity and size of plant fractal structures.",
            "units": "Count (integer)",
            "min": 1,
            "max": 10,
            "typical": 5,
            "default": 5
        }
    )

    # ======================
    # Adaptive Frame Management
    # ======================

    fps_low_threshold: float = field(
        default=60.0,
        metadata={
            "description": "FPS threshold below which culling begins to maintain performance.",
            "impact": "Triggers adaptive culling when FPS drops below this value to reduce particle count.",
            "units": "Frames per second",
            "min": 30.0,
            "max": 120.0,
            "typical": 60.0,
            "default": 60.0
        }
    )
    fps_high_threshold: float = field(
        default=240.0,
        metadata={
            "description": "FPS threshold above which energy boosting occurs to stimulate reproduction.",
            "impact": "Triggers energy boosts to cellular components when FPS exceeds this value, encouraging population growth.",
            "units": "Frames per second",
            "min": 60.0,
            "max": 300.0,
            "typical": 240.0,
            "default": 240.0
        }
    )
    cull_amount_per_frame: int = field(
        default=1,  # Prioritized to Version 2's default
        metadata={
            "description": "Number of particles to cull per frame when FPS is low.",
            "impact": "Defines the rate at which particles are removed to alleviate performance issues.",
            "units": "Count (integer)",
            "min": 1,
            "max": 100,
            "typical": 10,
            "default": 1  # Updated to match the field default
        }
    )
    energy_boost_factor: float = field(
        default=1.00,  # Prioritized to Version 2's default
        metadata={
            "description": "Factor by which to boost energy levels of all components when FPS is high.",
            "impact": "Enhances energy levels to promote reproduction and population growth.",
            "units": "Scalar (dimensionless)",
            "min": 1.0,
            "max": 2.0,
            "typical": 1.01,
            "default": 1.00  # Updated to match the field default
        }
    )
    min_total_particles: int = field(
        default=1000,
        metadata={
            "description": "Minimum total number of particles to maintain in the simulation.",
            "impact": "Ensures a baseline population size is preserved even during culling.",
            "units": "Count (integer)",
            "min": 500,
            "max": 10000,
            "typical": 1000,
            "default": 1000
        }
    )
    culling_ratio: float = field(
        default=0.01,
        metadata={
            "description": "Ratio determining how aggressively particles are culled relative to total population.",
            "impact": "Adjusts the proportion of particles to remove during culling to balance performance and population size.",
            "units": "Fraction (0.0 to 1.0)",
            "min": 0.0,
            "max": 0.1,
            "typical": 0.01,
            "default": 0.01
        }
    )

    # ======================
    # Culling Fitness Weights
    # ======================

    culling_fitness_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "energy_weight": 0.5,
            "age_weight": 1.0,
            "speed_factor_weight": 0.5,
            "interaction_strength_weight": 0.5
        },
        metadata={
            "description": "Weights determining the importance of various fitness factors during culling.",
            "impact": "Influences which particles are culled based on their energy, age, speed, and interaction strength.",
            "units": "Scalar (dimensionless)",
            "default": {
                "energy_weight": 0.5,
                "age_weight": 1.0,
                "speed_factor_weight": 0.5,
                "interaction_strength_weight": 0.5
            }
        }
    )

    # ======================
    # Day-Night Cycle
    # ======================

    day_night_cycle: Dict[str, Any] = field(
        default_factory=lambda: {
            "current_frame": 0,
            "is_day": True
        },
        metadata={
            "description": "State of the day-night cycle.",
            "impact": "Tracks the current phase (day or night) and frame count for transitions.",
            "units": "Dictionary with keys 'current_frame' and 'is_day'",
            "default": {
                "current_frame": 0,
                "is_day": True
            }
        }
    )

    # ======================
    # Initial Pause
    # ======================

    initial_pause_duration: int = field(
        default=1000,
        metadata={
            "description": "Duration of the initial pause before the simulation starts running.",
            "impact": "Allows for initial setup and stabilization before active simulation begins.",
            "units": "Milliseconds",
            "min": 0,
            "max": 10000,
            "typical": 1000,
            "default": 1000
        }
    )


import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


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
    # Filter out any non-vibrant colors
    vibrant_colors = [color for color in base_colors if color not in {"grey", "gray", "white", "black", "brown", "cream"}]
    random.shuffle(vibrant_colors)
    # Extend list if needed to ensure uniqueness
    while len(vibrant_colors) < n:
        unique_color = f"{random.choice(vibrant_colors)}_{random.randint(0, 999)}"
        if unique_color not in vibrant_colors:
            vibrant_colors.append(unique_color)
    return vibrant_colors[:n]


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
        Remove dead components, redistributing energy to neighbors.

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

            # Transfer energy from dead to alive neighbors
            for i, is_valid in enumerate(valid):
                if not any(is_valid):
                    continue
                valid_neighbor = neighbor_idxs[i][is_valid]
                num_neighbors = len(valid_neighbor)
                if num_neighbors > 0:
                    energy_per_neighbor = self.energy[dead_indices[i]] / num_neighbors
                    self.energy[alive_indices[valid_neighbor]] += energy_per_neighbor

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
# Food Particle Class
###############################################################

class FoodParticle:
    """
    Represents a stationary food particle providing energy to cellular components.

    Attributes:
    -----------
    x, y : float
        Coordinates in pixels.
    energy : float
        Energy value.
    alive : bool
        Availability status.
    """
    __slots__ = ['x', 'y', 'energy', 'alive']

    def __init__(self, x: float, y: float, energy: float = 50.0):
        """
        Initialize FoodParticle with position and energy.

        Parameters:
        -----------
        x, y : float
            Coordinates in pixels.
        energy : float, default=50.0
            Energy value.
        """
        self.x: float = x
        self.y: float = y
        self.energy: float = energy
        self.alive: bool = True

    def consume(self) -> float:
        """
        Consume the food particle, transferring its energy.

        Returns:
        --------
        float
            Transferred energy.
        """
        if not self.alive:
            return 0.0
        self.alive = False
        return self.energy


###############################################################
# Plant Class
###############################################################

class Plant:
    """
    Represents a plant with fractal growth patterns, producing food particles.

    Attributes:
    -----------
    x, y : float
        Coordinates in pixels.
    energy_efficiency : float
        Trait influencing energy consumption and growth.
    growth_stage : int
        Current growth stage.
    max_growth_stage : int
        Maximum growth stage.
    alive : bool
        Active status.
    gene_mutation_rate : float
        Mutation probability for traits.
    gene_mutation_range : Tuple[float, float]
        Mutation adjustment range.
    """

    def __init__(self, x: float, y: float, config: 'SimulationConfig'):
        """
        Initialize Plant with position and configuration.

        Parameters:
        -----------
        x, y : float
            Coordinates in pixels.
        config : SimulationConfig
            Simulation configuration instance.
        """
        self.x: float = x
        self.y: float = y
        self.energy_efficiency: float = np.random.uniform(
            config.energy_efficiency_range[0],
            config.energy_efficiency_range[1]
        )
        self.gene_mutation_rate: float = config.plant_mutation_rate
        self.gene_mutation_range: Tuple[float, float] = config.plant_mutation_range
        self.growth_stage: int = 0
        self.max_growth_stage: int = config.max_growth_stage
        self.alive: bool = True

    def grow(self, config: 'SimulationConfig') -> None:
        """
        Increment growth stage and produce food particles.

        Parameters:
        -----------
        config : SimulationConfig
            Simulation configuration instance.
        """
        if not self.alive or self.growth_stage >= self.max_growth_stage:
            return

        self.growth_stage += 1
        num_food = 2 ** self.growth_stage
        angle_increment = 360 / num_food

        angles = np.radians(np.arange(0, 360, angle_increment))
        distances = 50 * self.growth_stage
        food_x = self.x + distances * np.cos(angles)
        food_y = self.y + distances * np.sin(angles)
        energies = 25.0 * self.growth_stage

        # Assuming SimulationEnvironment is defined elsewhere with a class variable food_particles
        SimulationEnvironment.food_particles.extend(
            FoodParticle(x, y, energy=energy) for x, y, energy in zip(food_x, food_y, energies)
        )

    def mutate(self, config: 'SimulationConfig') -> None:
        """
        Apply mutations to energy efficiency trait.

        Parameters:
        -----------
        config : SimulationConfig
            Simulation configuration instance.
        """
        mutation_events = np.random.random() < self.gene_mutation_rate
        if mutation_events:
            mutation = random.uniform(*self.gene_mutation_range)
            self.energy_efficiency += mutation
            self.energy_efficiency = np.clip(
                self.energy_efficiency,
                config.energy_efficiency_range[0],
                config.energy_efficiency_range[1]
            )

    def update(self, config: 'SimulationConfig', current_frame: int) -> None:
        """
        Update plant state, handling growth and mutations.

        Parameters:
        -----------
        config : SimulationConfig
            Simulation configuration instance.
        current_frame : int
            Current frame count.
        """
        if config.enable_day_night_cycle:
            energy_modifier = config.plant_energy_modifier_day if config.day_night_cycle["is_day"] else config.plant_energy_modifier_night
            self.energy_efficiency *= energy_modifier
            self.energy_efficiency = np.clip(
                self.energy_efficiency,
                config.energy_efficiency_range[0],
                config.energy_efficiency_range[1]
            )

        # Determine growth frequency based on growth rate
        growth_frequency = max(1, int(100 / config.plant_growth_rate))
        if current_frame % growth_frequency == 0:
            self.grow(config)

        self.mutate(config)


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
    Manages interaction parameters between cellular type pairs.

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
        self.rules = self._create_interaction_matrix(mass_based_type_indices)
        self.give_take_matrix = self._create_give_take_matrix()
        self.synergy_matrix = self._create_synergy_matrix()

    def _create_interaction_matrix(self, mass_based_type_indices: List[int]) -> List[Tuple[int, int, Dict[str, Any]]]:
        """
        Construct interaction parameters for each type pair.

        Parameters:
        -----------
        mass_based_type_indices : List[int]
            Indices of mass-based types.

        Returns:
        --------
        List[Tuple[int, int, Dict[str, Any]]]
            List of interaction parameters.
        """
        interaction_rules = []
        for i in range(self.config.n_cell_types):
            for j in range(self.config.n_cell_types):
                if i == j:
                    continue
                params = self._random_interaction_params(i, j, mass_based_type_indices)
                interaction_rules.append((i, j, params))
        return interaction_rules

    def _random_interaction_params(self, i: int, j: int, mass_based_type_indices: List[int]) -> Dict[str, Any]:
        """
        Generate interaction parameters between two types.

        Parameters:
        -----------
        i, j : int
            Type indices.
        mass_based_type_indices : List[int]
            Indices of mass-based types.

        Returns:
        --------
        Dict[str, Any]
            Interaction parameters.
        """
        # Determine if both types are mass-based
        both_mass = (i in mass_based_type_indices and j in mass_based_type_indices)
        use_gravity = both_mass and (random.random() < 0.5)
        use_potential = True  # Potential is always used

        potential_strength = random.uniform(*self.config.interaction_strength_range)
        # Randomly invert potential strength to simulate attraction or repulsion
        if random.random() < 0.5:
            potential_strength = -potential_strength

        gravity_factor = random.uniform(0.1, 2.0) if use_gravity else 0.0
        max_dist = random.uniform(50.0, 200.0)

        return {
            "use_potential": use_potential,
            "use_gravity": use_gravity,
            "potential_strength": potential_strength,
            "gravity_factor": gravity_factor,
            "max_dist": max_dist
        }

    def _create_give_take_matrix(self) -> np.ndarray:
        """
        Create a matrix indicating give-take relationships.

        Returns:
        --------
        np.ndarray
            NxN boolean matrix.
        """
        matrix = np.zeros((self.config.n_cell_types, self.config.n_cell_types), dtype=bool)
        for i in range(self.config.n_cell_types):
            for j in range(self.config.n_cell_types):
                if i != j and random.random() < 0.1:
                    matrix[i, j] = True
        return matrix

    def _create_synergy_matrix(self) -> np.ndarray:
        """
        Create a matrix defining synergy factors.

        Returns:
        --------
        np.ndarray
            NxN float matrix.
        """
        synergy_matrix = np.zeros((self.config.n_cell_types, self.config.n_cell_types), dtype=float)
        for i in range(self.config.n_cell_types):
            for j in range(self.config.n_cell_types):
                if i != j and random.random() < 0.1:
                    synergy_matrix[i, j] = random.uniform(0.01, 0.3)
        return synergy_matrix


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
        Modify energy levels based on presence within zones.

        Parameters:
        -----------
        ct : CellularTypeData
            Cellular type data.
        """
        if not self.zones:
            return

        x = ct.x
        y = ct.y
        energy_modifiers = np.ones_like(x)

        for zone in self.zones:
            dx = x - zone["x_abs"]
            dy = y - zone["y_abs"]
            dist_sq = dx * dx + dy * dy
            in_zone = dist_sq <= zone["radius_sq"]
            energy_modifiers[in_zone] *= zone["effect"].get("energy_modifier", 1.0)

        ct.energy *= energy_modifiers


###############################################################
# Plant Manager
###############################################################

class PlantManager:
    """
    Manages all plant entities, handling growth, mutations, and rendering.

    Attributes:
    -----------
    config : SimulationConfig
        Simulation configuration.
    plants : List[Plant]
        Active plant entities.
    """
    def __init__(self, config: SimulationConfig):
        """
        Initialize PlantManager with configuration.

        Parameters:
        -----------
        config : SimulationConfig
            Simulation configuration instance.
        """
        self.config = config
        self.plants: List[Plant] = []

    def add_plant(self, plant: Plant) -> None:
        """
        Register a new plant.

        Parameters:
        -----------
        plant : Plant
            Plant to add.
        """
        self.plants.append(plant)

    def update_plants(self, current_frame: int) -> None:
        """
        Update all plants, handling growth and mutations.

        Parameters:
        -----------
        current_frame : int
            Current frame count.
        """
        if not self.config.enable_plants:
            return
        for plant in self.plants:
            plant.update(self.config, current_frame)

    def remove_dead_plants(self) -> None:
        """
        Remove inactive plants.
        """
        self.plants = [plant for plant in self.plants if plant.alive]

    def render_plants(self, renderer) -> None:
        """
        Delegate plant rendering to the Renderer.

        Parameters:
        -----------
        renderer
            Renderer instance.
        """
        if not self.config.enable_plants:
            return
        for plant in self.plants:
            self.draw_fractal(renderer, plant)

    def draw_fractal(self, renderer, plant: Plant) -> None:
        """
        Render a fractal structure representing a plant.

        Parameters:
        -----------
        renderer
            Renderer instance.
        plant : Plant
            Plant entity.
        """
        def draw_branch(x: float, y: float, length: float, angle: float, depth: int):
            if depth == 0:
                return
            end_x = x + length * math.cos(math.radians(angle))
            end_y = y + length * math.sin(math.radians(angle))
            pygame.gfxdraw.line(renderer.particle_surface, int(x), int(y), int(end_x), int(end_y), (34, 139, 34))
            draw_branch(end_x, end_y, length * 0.7, angle - 20, depth - 1)
            draw_branch(end_x, end_y, length * 0.7, angle + 20, depth - 1)

        initial_length = 50 * plant.energy_efficiency
        initial_angle = -90

        draw_branch(plant.x, plant.y, initial_length, initial_angle, plant.growth_stage)


###############################################################
# Forces & Interactions
###############################################################

def apply_interaction(a_x: float, a_y: float, b_x: float, b_y: float, params: Dict[str, Any]) -> Tuple[float, float]:
    """
    Calculate force exerted by B on A based on interaction parameters.

    Parameters:
    -----------
    a_x, a_y : float
        Coordinates of A.
    b_x, b_y : float
        Coordinates of B.
    params : Dict[str, Any]
        Interaction parameters.

    Returns:
    --------
    Tuple[float, float]
        Force components (fx, fy).
    """
    dx = a_x - b_x
    dy = a_y - b_y
    d_sq = dx * dx + dy * dy

    if d_sq == 0.0 or d_sq > params["max_dist"] ** 2:
        return 0.0, 0.0

    d = math.sqrt(d_sq)
    fx, fy = 0.0, 0.0

    if params.get("use_potential", True):
        F_pot = params.get("potential_strength", 1.0) / d
        fx += F_pot * dx
        fy += F_pot * dy

    if params.get("use_gravity", False):
        if "m_a" in params and "m_b" in params:
            F_grav = params.get("gravity_factor", 1.0) * (params["m_a"] * params["m_b"]) / d_sq
            fx += F_grav * dx
            fy += F_grav * dy

    return fx, fy

def give_take_interaction(giver_energy: float, receiver_energy: float,
                          giver_mass: Optional[float], receiver_mass: Optional[float],
                          config: SimulationConfig) -> Tuple[float, float, Optional[float], Optional[float]]:
    """
    Facilitate energy and mass transfer from giver to receiver.

    Parameters:
    -----------
    giver_energy : float
        Energy of giver.
    receiver_energy : float
        Energy of receiver.
    giver_mass : Optional[float]
        Mass of giver.
    receiver_mass : Optional[float]
        Mass of receiver.
    config : SimulationConfig
        Simulation configuration instance.

    Returns:
    --------
    Tuple[float, float, Optional[float], Optional[float]]
        Updated energies and masses.
    """
    transfer_amount = receiver_energy * config.energy_transfer_factor
    receiver_energy -= transfer_amount
    giver_energy += transfer_amount

    if config.mass_transfer and receiver_mass is not None and giver_mass is not None:
        mass_transfer = receiver_mass * config.energy_transfer_factor
        receiver_mass -= mass_transfer
        giver_mass += mass_transfer

    return giver_energy, receiver_energy, giver_mass, receiver_mass

def apply_clustering(ct: CellularTypeData, config: SimulationConfig) -> None:
    """
    Apply clustering forces for flocking behavior within a cellular type.

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

    tree = cKDTree(np.vstack((ct.x, ct.y)).T)
    neighbors = tree.query_ball_tree(tree, r=config.cluster_radius)

    dvx = np.zeros(n, dtype=float)
    dvy = np.zeros(n, dtype=float)

    for idx, neighbor_indices in enumerate(neighbors):
        neighbor_indices = [i for i in neighbor_indices if i != idx and ct.alive[i]]
        count = len(neighbor_indices)
        if count == 0:
            continue

        # Compute average velocity
        avg_vx = np.mean(ct.vx[neighbor_indices])
        avg_vy = np.mean(ct.vy[neighbor_indices])

        # Compute center of mass
        center_x = np.mean(ct.x[neighbor_indices])
        center_y = np.mean(ct.y[neighbor_indices])
        cohesion_vector_x = center_x - ct.x[idx]
        cohesion_vector_y = center_y - ct.y[idx]

        # Compute separation vector
        separation_vector_x = ct.x[idx] - np.mean(ct.x[neighbor_indices])
        separation_vector_y = ct.y[idx] - np.mean(ct.y[neighbor_indices])

        # Update velocity changes
        dvx[idx] += (config.alignment_strength * (avg_vx - ct.vx[idx]) +
                    config.cohesion_strength * cohesion_vector_x +
                    config.separation_strength * separation_vector_x)
        dvy[idx] += (config.alignment_strength * (avg_vy - ct.vy[idx]) +
                    config.cohesion_strength * cohesion_vector_y +
                    config.separation_strength * separation_vector_y)

    # Apply velocity changes
    ct.vx += dvx
    ct.vy += dvy

def apply_synergy(energyA: float, energyB: float, synergy_factor: float) -> Tuple[float, float]:
    """
    Balance energy between two allied components based on synergy factor.

    Parameters:
    -----------
    energyA : float
        Energy of A.
    energyB : float
        Energy of B.
    synergy_factor : float
        Energy sharing proportion.

    Returns:
    --------
    Tuple[float, float]
        Updated energies of A and B.
    """
    avg_energy = (energyA + energyB) * 0.5
    newA = (energyA * (1.0 - synergy_factor)) + (avg_energy * synergy_factor)
    newB = (energyB * (1.0 - synergy_factor)) + (avg_energy * synergy_factor)
    return newA, newB

def apply_colony_energy_sharing(ct: CellularTypeData, config: SimulationConfig) -> None:
    """
    Distribute energy among nearby components within a colony.

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

    tree = cKDTree(np.vstack((ct.x, ct.y)).T)
    sharing_radius = config.synergy_range
    neighbors = tree.query_ball_tree(tree, r=sharing_radius)

    energy_changes = np.zeros(n, dtype=float)

    for idx, neighbor_indices in enumerate(neighbors):
        for neighbor_idx in neighbor_indices:
            if neighbor_idx <= idx:
                continue
            avg_energy = (ct.energy[idx] + ct.energy[neighbor_idx]) / 2.0
            energy_diff = avg_energy - ct.energy[idx]
            energy_changes[idx] += energy_diff * config.synergy_matrix[idx, neighbor_idx]
            energy_changes[neighbor_idx] += -energy_diff * config.synergy_matrix[idx, neighbor_idx]

    ct.energy += energy_changes
    ct.energy = np.clip(ct.energy, 0.0, None)


###############################################################
# Renderer Class
###############################################################

class Renderer:
    """
    Handles rendering of components, plants, and food particles using optimized bulk operations with Pygame.

    Attributes:
    -----------
    surface : pygame.Surface
        Main Pygame surface.
    config : SimulationConfig
        Simulation configuration.
    color_map : Dict[str, Tuple[int, int, int]]
        Mapping of color names to RGB tuples.
    precomputed_colors : Dict[str, Tuple[int, int, int]]
        Precomputed color values for rapid access.
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

        self.precomputed_colors = {name: rgb for name, rgb in self.color_map.items()}

        # Initialize particle surface for double buffering
        self.particle_surface = pygame.Surface(self.surface.get_size(), flags=pygame.SRCALPHA)
        self.particle_surface = self.particle_surface.convert_alpha()

        # Initialize font for FPS display
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", self.config.font_size)

    def draw_components(self, ct: CellularTypeData) -> None:
        """
        Render all alive components of a cellular type using optimized bulk operations.

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

        # Initialize an empty pixel array
        pixel_array = np.zeros((self.surface.get_height(), self.surface.get_width(), 3), dtype=np.uint8)

        # Assign colors to pixel array
        pixel_array[y, x] = final_colors

        # Blit the pixel array onto the particle surface
        pygame.surfarray.blit_array(self.particle_surface, pixel_array.swapaxes(0, 1))

    def draw_food_particle(self, food: FoodParticle) -> None:
        """
        Render a single food particle.

        Parameters:
        -----------
        food : FoodParticle
            Food particle to render.
        """
        if not food.alive or not self.config.enable_food_particles:
            return

        pygame.gfxdraw.filled_circle(
            self.particle_surface,
            int(food.x),
            int(food.y),
            max(1, int(self.config.particle_size / 2)),
            (0, 255, 0)
        )
        pygame.gfxdraw.aacircle(
            self.particle_surface,
            int(food.x),
            int(food.y),
            max(1, int(self.config.particle_size / 2)),
            (0, 255, 0)
        )

    def render_food_particles(self) -> None:
        """
        Render all food particles.
        """
        if not self.config.enable_food_particles:
            return
        for food in SimulationEnvironment.food_particles:
            self.draw_food_particle(food)

    def render_plants(self, plant_manager: PlantManager) -> None:
        """
        Render all plants via PlantManager.

        Parameters:
        -----------
        plant_manager : PlantManager
            Manager handling plant entities.
        """
        plant_manager.render_plants(self)

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

    def render_boundary(self, bounded_space) -> None:
        """
        Render the boundary of the simulation area.

        Parameters:
        -----------
        bounded_space : BoundedSpace
            The bounded simulation area.
        """
        bounded_space.draw_boundary(self.surface)

    def render(self, is_day: bool, fps: float, bounded_space) -> None:
        """
        Composite rendered elements onto the main surface.

        Parameters:
        -----------
        is_day : bool
            Daytime status.
        fps : float
            Current FPS.
        bounded_space : BoundedSpace
            The bounded simulation area.
        """
        self.render_background(is_day)
        self.render_fps(fps)
        self.surface.blit(self.particle_surface, (0, 0))
        self.particle_surface.fill((0, 0, 0, 0))  # Clear particle surface for next frame
        self.render_boundary(bounded_space)


###############################################################
# Simulation Environment
###############################################################

class SimulationEnvironment:
    """
    Central environment managing global entities like food particles.
    """
    food_particles: List[FoodParticle] = []


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

        # Set window size
        window_width, window_height = 800, 600  # Default window size
        self.screen = pygame.display.set_mode((window_width, window_height), pygame.NOFRAME)
        pygame.display.set_caption("Ultra-Advanced Emergent Cellular Automata Simulation")

        self.clock = pygame.time.Clock()
        self.frame_count = 0
        self.run_flag = True
        
        self.edge_buffer = 0.01 * max(window_width, window_height)

        # Generate vibrant colors for cellular types
        self.colors = generate_vibrant_colors(self.config.n_cell_types)
        n_mass_types = int(self.config.mass_based_fraction * self.config.n_cell_types)
        mass_based_type_indices = list(range(n_mass_types))

        # Initialize CellularTypeManager
        self.type_manager = CellularTypeManager(self.config, self.colors, mass_based_type_indices)

        # Initialize cellular types
        for i in range(self.config.n_cell_types):
            if i < n_mass_types:
                mass_val = random.uniform(*self.config.mass_range)
                ct = CellularTypeData(
                    type_id=i,
                    color=self.colors[i],
                    n_particles=self.config.particles_per_type,
                    window_width=window_width,
                    window_height=window_height,
                    config=self.config,
                    mass=mass_val,
                    base_velocity_scale=self.config.base_velocity_scale
                )
            else:
                ct = CellularTypeData(
                    type_id=i,
                    color=self.colors[i],
                    n_particles=self.config.particles_per_type,
                    window_width=window_width,
                    window_height=window_height,
                    config=self.config,
                    mass=None,
                    base_velocity_scale=self.config.base_velocity_scale
                )
            self.type_manager.add_cellular_type_data(ct)

        # Initialize InteractionRules
        self.rules_manager = InteractionRules(self.config, mass_based_type_indices)

        # Initialize EnvironmentalManager
        self.env_manager = EnvironmentalManager(self.config, window_width, window_height)

        # Initialize PlantManager
        self.plant_manager = PlantManager(self.config)

        # Initialize Renderer
        self.renderer = Renderer(self.screen, self.config)

        # Initialize Day-Night Cycle
        self.day_night_cycle = self.config.day_night_cycle.copy()

        # Initialize Plants
        self.initialize_plants(window_width, window_height)

        self.protected_colonies: List[int] = []

        # Initialize Paused State
        self.paused = True
        self.pause_start_time = pygame.time.get_ticks()
        logging.info("Simulation initialized and paused for initial setup.")
        self.cull_factor: float = 1.0  # Initial culling factor

    def initialize_plants(self, screen_width: int, screen_height: int) -> None:
        """
        Seed simulation with initial plant entities.

        Parameters:
        -----------
        screen_width, screen_height : int
            Window dimensions.
        """
        if not self.config.enable_plants:
            return

        # Central Plant
        central_plant = Plant(screen_width / 2, screen_height / 2, self.config)
        self.plant_manager.add_plant(central_plant)

        # Additional Random Plants
        for _ in range(int(self.config.n_cell_types * 0.1)):
            x, y = random_xy(screen_width, screen_height, 1)[0]
            plant = Plant(x, y, self.config)
            self.plant_manager.add_plant(plant)

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

            current_time = pygame.time.get_ticks()
            elapsed_time = current_time - self.pause_start_time

            if self.paused and elapsed_time >= self.config.initial_pause_duration:
                self.paused = False
                logging.info("Initial setup complete. Simulation is now running.")

            if not self.paused:
                self.update_day_night_cycle()
                self.apply_all_interactions()
                for ct in self.type_manager.cellular_types:
                    self.env_manager.apply_zone_effects(ct)
                for ct in self.type_manager.cellular_types:
                    apply_clustering(ct, self.config)
                self.type_manager.reproduce()
                self.plant_manager.update_plants(self.frame_count)
                self.type_manager.remove_dead_in_all_types()
                self.plant_manager.remove_dead_plants()
                self.consume_food_particles()
                for ct in self.type_manager.cellular_types:
                    self.renderer.draw_components(ct)
                self.renderer.render_plants(self.plant_manager)
                self.renderer.render_food_particles()
                self.handle_boundary_reflections()

            is_day = self.day_night_cycle["is_day"]

            current_fps = self.clock.get_fps()
            fps_history.append(current_fps)
            if len(fps_history) > fps_history_length:
                fps_history.pop(0)
            average_fps = np.mean(fps_history) if fps_history else current_fps

            self.renderer.render(is_day, average_fps, BoundedSpace(self.screen.get_width(), self.screen.get_height(), self.config))
            pygame.display.flip()
            self.clock.tick(60)  # Standardized to 60 FPS for consistency

            if not self.paused:
                self.adaptive_culling(average_fps)

    def update_day_night_cycle(self) -> None:
        """
        Manage progression of day-night cycle, toggling based on frame counts.
        """
        if not self.config.enable_day_night_cycle:
            return

        self.day_night_cycle["current_frame"] += 1

        if self.day_night_cycle["is_day"]:
            if self.day_night_cycle["current_frame"] >= self.config.day_duration:
                self.day_night_cycle["is_day"] = False
                self.day_night_cycle["current_frame"] = 0
                logging.info("Transitioning to night.")
        else:
            if self.day_night_cycle["current_frame"] >= self.config.night_duration:
                self.day_night_cycle["is_day"] = True
                self.day_night_cycle["current_frame"] = 0
                logging.info("Transitioning to day.")

    def apply_all_interactions(self) -> None:
        """
        Apply all defined interactions between cellular types.
        """
        for (i, j, params) in self.rules_manager.rules:
            self.apply_interaction_between_types(i, j, params)

    def apply_interaction_between_types(self, i: int, j: int, params: Dict[str, Any]) -> None:
        """
        Apply interaction parameters between two cellular types.

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

        if params.get("use_gravity", False):
            if ct_i.mass_based and ct_j.mass_based and ct_i.mass is not None and ct_j.mass is not None:
                params["m_a"] = ct_i.mass
                params["m_b"] = ct_j.mass
            else:
                params["use_gravity"] = False

        # Efficient distance computation using broadcasting
        dx = ct_i.x[:, np.newaxis] - ct_j.x[np.newaxis, :]
        dy = ct_i.y[:, np.newaxis] - ct_j.y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        within_range = (dist_sq > 0.0) & (dist_sq <= params["max_dist"]**2)
        indices = np.where(within_range)

        if len(indices[0]) == 0:
            return

        dist = np.sqrt(dist_sq[indices])
        fx = np.zeros_like(dist)
        fy = np.zeros_like(dist)

        if params.get("use_potential", True):
            F_pot = params.get("potential_strength", 1.0) / dist
            fx += F_pot * dx[indices]
            fy += F_pot * dy[indices]

        if params.get("use_gravity", False):
            F_grav = params.get("gravity_factor", 1.0) * (params["m_a"][indices[0]] * params["m_b"][indices[1]]) / dist_sq[indices]
            fx += F_grav * dx[indices]
            fy += F_grav * dy[indices]

        # Update velocities using NumPy's advanced indexing
        np.add.at(ct_i.vx, indices[0], fx)
        np.add.at(ct_i.vy, indices[0], fy)

        # Handle give-take interactions
        if is_giver and self.config.enable_culling:
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

        # Apply friction and global temperature
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
            left = self.edge_buffer
            right = self.screen.get_width() - self.edge_buffer
            close_left = ct.x < left
            close_right = ct.x > right
            ct.vx[close_left | close_right] *= -1
            ct.x = np.clip(ct.x, left, right)

            # Reflect off top and bottom edges
            top = self.edge_buffer
            bottom = self.screen.get_height() - self.edge_buffer
            close_top = ct.y < top
            close_bottom = ct.y > bottom
            ct.vy[close_top | close_bottom] *= -1
            ct.y = np.clip(ct.y, top, bottom)

    def handle_boundary_reflection(self, ct: CellularTypeData) -> None:
        """
        Reflect cellular components of a single type off window boundaries.

        Parameters:
        -----------
        ct : CellularTypeData
            Cellular type data.
        """
        if ct.x.size == 0:
            return

        # Reflect off left and right edges
        left = self.edge_buffer
        right = self.screen.get_width() - self.edge_buffer
        close_left = ct.x < left
        close_right = ct.x > right
        ct.vx[close_left | close_right] *= -1
        ct.x = np.clip(ct.x, left, right)

        # Reflect off top and bottom edges
        top = self.edge_buffer
        bottom = self.screen.get_height() - self.edge_buffer
        close_top = ct.y < top
        close_bottom = ct.y > bottom
        ct.vy[close_top | close_bottom] *= -1
        ct.y = np.clip(ct.y, top, bottom)

    def consume_food_particles(self) -> None:
        """
        Allow cellular components to consume nearby food particles, transferring energy.
        """
        if not self.config.enable_food_particles:
            return

        consumption_radius = 10.0
        consumption_radius_sq = consumption_radius ** 2

        # Precompute alive food particles
        alive_food = [food for food in SimulationEnvironment.food_particles if food.alive]
        if not alive_food:
            return

        food_positions = np.array([[food.x, food.y] for food in alive_food])
        if food_positions.size == 0:
            return

        # Use cKDTree for efficient proximity queries
        food_tree = cKDTree(food_positions)

        for ct in self.type_manager.cellular_types:
            if ct.x.size == 0:
                continue

            # Query for food within consumption radius
            component_coords = np.vstack((ct.x, ct.y)).T
            indices = food_tree.query_ball_point(component_coords, r=consumption_radius)

            # Iterate over components and consume the nearest food if available
            for component_idx, food_indices in enumerate(indices):
                if not food_indices:
                    continue
                # Consume the first available food
                food_idx = food_indices[0]
                food = alive_food[food_idx]
                energy_gained = food.consume()
                if energy_gained > 0.0:
                    ct.energy[component_idx] += energy_gained
                    # Remove the consumed food from the environment
                    last_food = SimulationEnvironment.food_particles[-1]
                    SimulationEnvironment.food_particles[food_idx] = last_food
                    SimulationEnvironment.food_particles.pop()
                    alive_food.pop(food_idx)  # Remove from local list

    def adaptive_culling(self, average_fps: float) -> None:
        """
        Perform dynamic culling based on current FPS to maintain performance.

        Parameters:
        -----------
        average_fps : float
            Current average FPS.
        """
        if not self.config.enable_culling:
            return

        if average_fps < self.config.fps_low_threshold:
            incremental_cull = int(self.config.cull_amount_per_frame * self.cull_factor)
            self.intelligent_cull(incremental_cull)
            self.cull_factor *= 1.1  # Increase culling factor for next frame
        elif average_fps > self.config.fps_high_threshold:
            self.cull_factor = max(1.0, self.cull_factor * 0.9)  # Decrease if FPS improves

    def intelligent_cull(self, num_to_cull: int) -> None:
        """
        Cull a specified number of particles based on fitness without prioritizing any type.

        Parameters:
        -----------
        num_to_cull : int
            Number of particles to cull.
        """
        total_particles = sum(ct.x.size for ct in self.type_manager.cellular_types)
        excess_particles = total_particles - self.config.min_total_particles
        if excess_particles <= 0:
            return

        num_to_cull = min(num_to_cull, excess_particles)

        fitness_data = []
        for ct in self.type_manager.cellular_types:
            if ct.x.size == 0:
                continue
            fitness = (
                self.config.culling_fitness_weights["energy_weight"] * ct.energy +
                self.config.culling_fitness_weights["age_weight"] * (1.0 / (ct.age + 1.0)) +
                self.config.culling_fitness_weights["speed_factor_weight"] * ct.speed_factor +
                self.config.culling_fitness_weights["interaction_strength_weight"] * ct.interaction_strength
            )
            fitness_data.append((ct, fitness))

        all_fitness = np.concatenate([data[1] for data in fitness_data])
        if len(all_fitness) == 0:
            return

        # Identify indices with lowest fitness
        cull_indices = np.argsort(all_fitness)[:num_to_cull]

        cumulative = 0
        for ct, fitness in fitness_data:
            if cumulative >= num_to_cull:
                break
            ct_num = ct.x.size
            cull_count = min(num_to_cull - cumulative, ct_num)
            cull_subset = np.argsort(fitness)[:cull_count]

            # Apply culling
            mask = np.ones(ct.x.size, dtype=bool)
            mask[cull_subset] = False

            ct.x = ct.x[mask]
            ct.y = ct.y[mask]
            ct.vx = ct.vx[mask]
            ct.vy = ct.vy[mask]
            ct.energy = ct.energy[mask]
            ct.alive = ct.alive[mask]
            ct.age = ct.age[mask]
            ct.energy_efficiency = ct.energy_efficiency[mask]
            ct.speed_factor = ct.speed_factor[mask]
            ct.interaction_strength = ct.interaction_strength[mask]
            if ct.mass_based and ct.mass is not None:
                ct.mass = ct.mass[mask]

            cumulative += cull_count

        logging.info(f"Culled {cumulative} particles to maintain performance.")

    def run(self) -> None:
        """
        Start and manage the simulation loop.
        """
        self.main_loop()

###############################################################
# Bounded Space Class
###############################################################

@dataclass
class BoundedSpace:
    """
    Class to define and manage the bounded simulation area within the window.

    Attributes:
    -----------
    left, right, top, bottom : int
        Boundaries of the simulation area.
    width, height : int
        Dimensions of the simulation area.
    rect : pygame.Rect
        Pygame rectangle representing the bounded area.
    """
    left: int
    right: int
    top: int
    bottom: int
    width: int
    height: int
    rect: pygame.Rect

    def __init__(self, screen_width: int, screen_height: int, config: SimulationConfig):
        """
        Initialize the bounded space based on screen dimensions and configuration.

        Parameters:
        -----------
        screen_width (int): Width of the window in pixels.
        screen_height (int): Height of the window in pixels.
        config (SimulationConfig): Configuration settings for margins.
        """
        self.left = config.edge_buffer
        self.right = screen_width - config.edge_buffer
        self.top = config.edge_buffer
        self.bottom = screen_height - config.edge_buffer
        self.width = self.right - self.left
        self.height = self.bottom - self.top
        self.rect = pygame.Rect(self.left, self.top, self.width, self.height)
        self._validate_boundaries()

    def _validate_boundaries(self):
        """
        Ensure that the bounded space dimensions are positive.
        """
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Bounded space dimensions must be positive. Check margin settings in the configuration.")

    def draw_boundary(self, surface: pygame.Surface, color: Tuple[int, int, int] = (255, 255, 255),
                      thickness: int = 2):
        """
        Draw the boundary of the simulation area on the given surface.

        Parameters:
        -----------
        surface (pygame.Surface): The surface to draw the boundary on.
        color (Tuple[int, int, int], optional): RGB color of the boundary. Defaults to white.
        thickness (int, optional): Thickness of the boundary lines. Defaults to 2.
        """
        pygame.draw.rect(surface, color, self.rect, thickness)


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
