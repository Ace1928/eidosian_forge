import asyncio
import random
import sys
from concurrent.futures import ThreadPoolExecutor
import functools
from functools import lru_cache

import duckdb
import numpy as np
import pygame
from numba import jit, njit, prange

# Initialize Pygame and set up the display
pygame.init()
display_info = pygame.display.Info()
width, height = display_info.current_w - 100, display_info.current_h - 100
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
center = np.array([width // 2, height // 2])

# Constants
GRAVITY_STRENGTH = 0.00005
CHARGE_CONSTANT = 0.00001
QUANTUM_FORCE_CONSTANT = 0.001
NUCLEAR_FORCE_RANGE = 4
NUCLEAR_FORCE_STRENGTH = 0.002
DARK_MATTER_PULL = 0.03
PARTICLE_COUNT = 300
PARTICLE_SIZE = 2
EPSILON = 1e-11
PHOTON_ENERGY_FACTOR = 0.001
BINDING_ENERGY_FACTOR = 0.0005
MAX_ELEMENT = "iron"
TARGET_FPS = 60  # Desired frame rate for the simulation
DT = 1 / TARGET_FPS  # Time step based on the desired frame rate
FUSION_THRESHOLD = 10
REPULSIVE_CONSTANT = 0.0001

# Initialize DuckDB for particle data
conn = duckdb.connect(database=":memory:", read_only=False)
conn.execute(
    """
    CREATE TABLE particles (
        id INTEGER,
        x FLOAT,
        y FLOAT,
        vx FLOAT,
        vy FLOAT,
        mass FLOAT,
        element VARCHAR,
        color VARCHAR,
        em_charge INTEGER,
        em_radius FLOAT,
        strong_coupling FLOAT,
        strong_radius FLOAT
    )
    """
)

# Periodic table simulation
ELEMENTS = {
    "hydrogen": {
        "mass": 1.008,
        "color": (255, 255, 255),
        "fusion_products": [
            {"element": "helium", "probability": 1.0, "photon_energy": 0.0015}
        ],
        "em_charge": 1,
        "em_radius": 0.53,
        "strong_coupling": 0.1,
        "strong_radius": 0.01,
    },
    "helium": {
        "mass": 4.002602,
        "color": (217, 255, 0),
        "fusion_products": [
            {"element": "beryllium", "probability": 0.9, "photon_energy": 0.003},
            {"element": "carbon", "probability": 0.1, "photon_energy": 0.006},
        ],
        "em_charge": 0,
        "em_radius": 0.31,
        "strong_coupling": 0.2,
        "strong_radius": 1.2,
    },
    "lithium": {
        "mass": 6.941,
        "color": (204, 128, 255),
        "fusion_products": [
            {"element": "beryllium", "probability": 1.0, "photon_energy": 0.0045}
        ],
        "em_charge": 1,
        "em_radius": 1.45,
        "strong_coupling": 0.3,
        "strong_radius": 1.7,
    },
    "beryllium": {
        "mass": 9.0122,
        "color": (76, 255, 0),
        "fusion_products": [
            {"element": "carbon", "probability": 1.0, "photon_energy": 0.0075}
        ],
        "em_charge": 2,
        "em_radius": 1.12,
        "strong_coupling": 0.4,
        "strong_radius": 1.9,
    },
    "boron": {
        "mass": 10.811,
        "color": (0, 255, 0),
        "fusion_products": [
            {"element": "carbon", "probability": 1.0, "photon_energy": 0.009}
        ],
        "em_charge": 3,
        "em_radius": 0.9,
        "strong_coupling": 0.5,
        "strong_radius": 2.1,
    },
    "carbon": {
        "mass": 12.011,
        "color": (0, 255, 111),
        "fusion_products": [
            {"element": "oxygen", "probability": 0.8, "photon_energy": 0.0105},
            {"element": "neon", "probability": 0.2, "photon_energy": 0.0135},
        ],
        "em_charge": 4,
        "em_radius": 0.77,
        "strong_coupling": 0.6,
        "strong_radius": 2.3,
    },
    "nitrogen": {
        "mass": 14.007,
        "color": (0, 255, 255),
        "fusion_products": [
            {"element": "oxygen", "probability": 1.0, "photon_energy": 0.012}
        ],
        "em_charge": -3,
        "em_radius": 0.7,
        "strong_coupling": 0.7,
        "strong_radius": 2.5,
    },
    "oxygen": {
        "mass": 15.999,
        "color": (0, 111, 255),
        "fusion_products": [
            {"element": "neon", "probability": 0.9, "photon_energy": 0.015},
            {"element": "magnesium", "probability": 0.1, "photon_energy": 0.0165},
        ],
        "em_charge": -2,
        "em_radius": 0.66,
        "strong_coupling": 0.8,
        "strong_radius": 2.7,
    },
    "fluorine": {
        "mass": 18.998,
        "color": (89, 0, 255),
        "fusion_products": [
            {"element": "neon", "probability": 1.0, "photon_energy": 0.018}
        ],
        "em_charge": -1,
        "em_radius": 0.64,
        "strong_coupling": 0.9,
        "strong_radius": 2.9,
    },
    "neon": {
        "mass": 20.180,
        "color": (143, 0, 255),
        "fusion_products": [
            {"element": "magnesium", "probability": 1.0, "photon_energy": 0.0195}
        ],
        "em_charge": 0,
        "em_radius": 0.58,
        "strong_coupling": 1.0,
        "strong_radius": 3.1,
    },
    "sodium": {
        "mass": 22.990,
        "color": (255, 217, 0),
        "fusion_products": [
            {"element": "magnesium", "probability": 1.0, "photon_energy": 0.02025}
        ],
        "em_charge": 1,
        "em_radius": 1.9,
        "strong_coupling": 1.1,
        "strong_radius": 3.3,
    },
    "magnesium": {
        "mass": 24.305,
        "color": (255, 0, 217),
        "fusion_products": [
            {"element": "silicon", "probability": 1.0, "photon_energy": 0.021}
        ],
        "em_charge": 2,
        "em_radius": 1.6,
        "strong_coupling": 1.2,
        "strong_radius": 3.5,
    },
    "aluminum": {
        "mass": 26.982,
        "color": (191, 191, 191),
        "fusion_products": [
            {"element": "silicon", "probability": 1.0, "photon_energy": 0.02175}
        ],
        "em_charge": 3,
        "em_radius": 1.43,
        "strong_coupling": 1.3,
        "strong_radius": 3.7,
    },
    "silicon": {
        "mass": 28.085,
        "color": (255, 0, 76),
        "fusion_products": [
            {"element": "sulfur", "probability": 0.9, "photon_energy": 0.0225},
            {"element": "argon", "probability": 0.1, "photon_energy": 0.024},
        ],
        "em_charge": 4,
        "em_radius": 1.32,
        "strong_coupling": 1.4,
        "strong_radius": 3.9,
    },
    "phosphorus": {
        "mass": 30.974,
        "color": (255, 128, 0),
        "fusion_products": [
            {"element": "sulfur", "probability": 1.0, "photon_energy": 0.02325}
        ],
        "em_charge": -3,
        "em_radius": 1.28,
        "strong_coupling": 1.5,
        "strong_radius": 4.1,
    },
    "sulfur": {
        "mass": 32.06,
        "color": (255, 255, 0),
        "fusion_products": [
            {"element": "argon", "probability": 0.9, "photon_energy": 0.024},
            {"element": "calcium", "probability": 0.1, "photon_energy": 0.0255},
        ],
        "em_charge": -2,
        "em_radius": 1.27,
        "strong_coupling": 1.6,
        "strong_radius": 4.3,
    },
    "chlorine": {
        "mass": 35.45,
        "color": (0, 255, 0),
        "fusion_products": [
            {"element": "argon", "probability": 1.0, "photon_energy": 0.02475}
        ],
        "em_charge": -1,
        "em_radius": 1.27,
        "strong_coupling": 1.7,
        "strong_radius": 4.5,
    },
    "argon": {
        "mass": 39.948,
        "color": (128, 255, 255),
        "fusion_products": [
            {"element": "calcium", "probability": 1.0, "photon_energy": 0.0255}
        ],
        "em_charge": 0,
        "em_radius": 1.1,
        "strong_coupling": 1.8,
        "strong_radius": 4.7,
    },
    "potassium": {
        "mass": 39.098,
        "color": (255, 0, 255),
        "fusion_products": [
            {"element": "calcium", "probability": 1.0, "photon_energy": 0.02625}
        ],
        "em_charge": 1,
        "em_radius": 2.43,
        "strong_coupling": 1.9,
        "strong_radius": 4.9,
    },
    "calcium": {
        "mass": 40.078,
        "color": (255, 255, 255),
        "fusion_products": [
            {"element": "titanium", "probability": 0.9, "photon_energy": 0.027},
            {"element": "chromium", "probability": 0.1, "photon_energy": 0.0285},
        ],
        "em_charge": 2,
        "em_radius": 1.97,
        "strong_coupling": 2.0,
        "strong_radius": 5.1,
    },
    "scandium": {
        "mass": 44.956,
        "color": (255, 128, 128),
        "fusion_products": [
            {"element": "titanium", "probability": 1.0, "photon_energy": 0.02775}
        ],
        "em_charge": 3,
        "em_radius": 1.62,
        "strong_coupling": 2.1,
        "strong_radius": 5.3,
    },
    "titanium": {
        "mass": 47.867,
        "color": (191, 191, 191),
        "fusion_products": [
            {"element": "chromium", "probability": 0.9, "photon_energy": 0.0285},
            {"element": "iron", "probability": 0.1, "photon_energy": 0.03},
        ],
        "em_charge": 4,
        "em_radius": 1.47,
        "strong_coupling": 2.2,
        "strong_radius": 5.5,
    },
    "vanadium": {
        "mass": 50.942,
        "color": (128, 128, 255),
        "fusion_products": [
            {"element": "chromium", "probability": 1.0, "photon_energy": 0.02925}
        ],
        "em_charge": 5,
        "em_radius": 1.34,
        "strong_coupling": 2.3,
        "strong_radius": 5.7,
    },
    "chromium": {
        "mass": 51.996,
        "color": (128, 128, 128),
        "fusion_products": [
            {"element": "iron", "probability": 1.0, "photon_energy": 0.03}
        ],
        "em_charge": 6,
        "em_radius": 1.27,
        "strong_coupling": 2.4,
        "strong_radius": 5.9,
    },
    "manganese": {
        "mass": 54.938,
        "color": (128, 128, 128),
        "fusion_products": [
            {"element": "iron", "probability": 1.0, "photon_energy": 0.03075}
        ],
        "em_charge": 7,
        "em_radius": 1.26,
        "strong_coupling": 2.5,
        "strong_radius": 6.1,
    },
    "iron": {
        "mass": 55.845,
        "color": (128, 128, 128),
        "fusion_products": [],
        "em_charge": 0,
        "em_radius": 1.26,
        "strong_coupling": 2.6,
        "strong_radius": 6.3,
    },
}


class Particle:
    """
    A class representing a particle in the simulation.

    Attributes:
        id (int): Unique identifier for the particle.
        pos (np.ndarray): Position of the particle in 2D space.
        vel (np.ndarray): Velocity of the particle in 2D space.
        element (str): Element type of the particle.
        properties (dict): Properties of the particle based on its element.
        mass (float): Mass of the particle.
        color (tuple): Color of the particle for rendering.
        em_charge (int): Electromagnetic charge of the particle.
        em_radius (float): Electromagnetic radius of the particle.
        strong_coupling (float): Strong nuclear force coupling constant of the particle.
        strong_radius (float): Strong nuclear force radius of the particle.
    """

    def __init__(
        self, x: float, y: float, vx: float, vy: float, element: str = "hydrogen"
    ):
        """
        Initialize a Particle instance.

        Args:
            x (float): x-coordinate of the particle's initial position.
            y (float): y-coordinate of the particle's initial position.
            vx (float): x-component of the particle's initial velocity.
            vy (float): y-component of the particle's initial velocity.
            element (str): Element type of the particle (default: "hydrogen").
        """
        self.id = None  # Unique identifier for the particle
        self.pos = np.array(
            [x, y], dtype=np.float64
        )  # Position of the particle in 2D space
        self.vel = np.array(
            [vx, vy], dtype=np.float64
        )  # Velocity of the particle in 2D space
        self.element = element  # Element type of the particle
        self.properties = ELEMENTS[
            element
        ]  # Properties of the particle based on its element
        self.mass = self.properties["mass"]  # Mass of the particle
        self.color = self.properties["color"]  # Color of the particle for rendering
        self.em_charge = self.properties[
            "em_charge"
        ]  # Electromagnetic charge of the particle
        self.em_radius = self.properties[
            "em_radius"
        ]  # Electromagnetic radius of the particle
        self.strong_coupling = self.properties[
            "strong_coupling"
        ]  # Strong nuclear force coupling constant
        self.strong_radius = self.properties[
            "strong_radius"
        ]  # Strong nuclear force radius

    @functools.lru_cache(maxsize=1024)
    @njit
    def calculate_force(self, other: "Particle") -> np.ndarray:
        """
        Calculate the force exerted by another particle on this particle.

        Args:
            other (Particle): The other particle.

        Returns:
            np.ndarray: The force vector.
        """
        displacement = other.pos - self.pos
        distance_sq = np.dot(displacement, displacement)
        distance = np.sqrt(distance_sq) + EPSILON

        force = np.zeros(2, dtype=np.float64)

        # Gravitational force
        gravitational_force = displacement * (
            GRAVITY_STRENGTH * self.mass * other.mass / (distance_sq * distance)
        )
        force += gravitational_force

        # Electromagnetic force
        em_force_magnitude = (
            CHARGE_CONSTANT
            * self.em_charge
            * other.em_charge
            / (distance_sq + (self.em_radius + other.em_radius) ** 2)
        )
        electromagnetic_force = displacement * em_force_magnitude
        force += electromagnetic_force

        # Strong nuclear force
        if distance < self.strong_radius + other.strong_radius:
            strong_force_magnitude = (
                NUCLEAR_FORCE_STRENGTH
                * self.strong_coupling
                * other.strong_coupling
                / distance_sq
            )
            strong_nuclear_force = displacement * strong_force_magnitude
            force -= strong_nuclear_force

        # Repulsive force (simplified Coulomb's Law)
        repulsive_force_magnitude = (
            self.em_charge * other.em_charge * REPULSIVE_CONSTANT / distance_sq
        )
        repulsive_force = displacement * repulsive_force_magnitude / distance
        force += repulsive_force

        return force

    def apply_forces(self, particles: np.ndarray):
        """
        Apply forces from other particles, dark matter, and quantum fluctuations to update the particle's velocity and position.

        Args:
            particles (np.ndarray): Array of all particles in the simulation.
        """
        force = np.zeros(2, dtype=np.float64)

        # Efficiently calculate and accumulate the force from each other particle using numpy array operations
        ids = particles["id"]
        mask = ids != self.id
        other_particles = particles[mask]
        forces = np.array([self.calculate_force(other) for other in other_particles])
        force += np.sum(forces, axis=0)

        # Calculate the direction vector from the particle to the center of the simulation
        direction_to_center = center - self.pos
        # Apply the dark matter force
        force += direction_to_center * DARK_MATTER_PULL

        # Apply random quantum fluctuations to the force
        force += np.random.randn(2) * QUANTUM_FORCE_CONSTANT

        # Update the particle's velocity based on the total force and timestep
        self.vel += force / self.mass * DT
        # Update the particle's position using Verlet integration
        self.pos += self.vel * DT + 0.5 * force / self.mass * DT**2

        # Ensure the particle stays within the simulation boundaries and reflects if it hits the boundary
        old_pos = np.copy(self.pos)
        self.pos = np.clip(self.pos, 0, [width, height])
        # Reflect the velocity vector if the particle hits the boundary
        if np.any(self.pos == 0) or np.any(self.pos == [width, height]):
            reflection_mask = old_pos == self.pos
            self.vel[reflection_mask] = -self.vel[reflection_mask]

    def draw(self):
        """
        Draw the particle on the screen.
        """
        radius = max(min(30, PARTICLE_SIZE * (self.properties["mass"] / 10)), 2)
        pygame.draw.circle(screen, self.color, self.pos.astype(int), radius)
        # Draw the particle as a circle on the screen

    @functools.lru_cache(maxsize=1024)
    def emit_photon(self):
        """
        Emit a photon from the particle, reducing its mass.

        Returns:
            float: The energy of the emitted photon.
        """
        energy_loss = (
            self.mass * PHOTON_ENERGY_FACTOR
        )  # Calculate the energy loss based on the particle's mass and a constant factor
        self.mass -= energy_loss  # Reduce the particle's mass by the energy loss
        return energy_loss  # Return the energy of the emitted photon

    @functools.lru_cache(maxsize=1024)
    def fusion(self, other):
        """
        Perform fusion with another particle if the conditions are met.

        Args:
            other (Particle): The other particle.

        Returns:
            float: The energy of the photon emitted during fusion, or 0 if no fusion occurred.
        """
        relative_velocity = np.linalg.norm(self.vel - other.vel)
        collision_energy = 0.5 * self.mass * relative_velocity**2

        if collision_energy > FUSION_THRESHOLD:
            if (
                np.sum((self.pos - other.pos) ** 2)
                < (self.strong_radius + other.strong_radius) ** 2
            ):
                fusion_products = self.properties[
                    "fusion_products"
                ]  # Get the fusion products for the particle's element
                if fusion_products:
                    new_element, probability, photon_energy = max(
                        fusion_products, key=lambda x: x["probability"]
                    ).values()  # Get the fusion product with the highest probability

                    if (
                        random.random() < probability
                    ):  # Check if fusion occurs based on the probability
                        self.mass += (
                            other.mass - BINDING_ENERGY_FACTOR
                        )  # Update the particle's mass by adding the other particle's mass and subtracting the binding energy
                        self.element = new_element  # Update the particle's element to the new element after fusion
                        self.properties = ELEMENTS[
                            new_element
                        ]  # Update the particle's properties based on the new element
                        self.color = self.properties[
                            "color"
                        ]  # Update the particle's color
                        self.em_charge = self.properties[
                            "em_charge"
                        ]  # Update the particle's electromagnetic charge
                        self.em_radius = self.properties[
                            "em_radius"
                        ]  # Update the particle's electromagnetic radius
                        self.strong_coupling = self.properties[
                            "strong_coupling"
                        ]  # Update the particle's strong nuclear force coupling constant
                        self.strong_radius = self.properties[
                            "strong_radius"
                        ]  # Update the particle's strong nuclear force radius
                        other.mass = 0  # Set the other particle's mass to zero (consumed during fusion)
                        return photon_energy  # Return the energy of the photon emitted during fusion

        return 0  # Return 0 if no fusion occurred


@functools.lru_cache(maxsize=1024)
def create_particle(x, y, vx, vy, element):
    """
    Create a new particle and insert its data into the database.

    Args:
        x (float): x-coordinate of the particle's initial position.
        y (float): y-coordinate of the particle's initial position.
        vx (float): x-component of the particle's initial velocity.
        vy (float): y-component of the particle's initial velocity.
        element (str): Element type of the particle.

    Returns:
        Particle: The created particle instance.
    """
    particle = Particle(x, y, vx, vy, element)  # Create a new particle instance
    particle_data = (
        particle.pos[0],  # x-coordinate of the particle's position
        particle.pos[1],  # y-coordinate of the particle's position
        particle.vel[0],  # x-component of the particle's velocity
        particle.vel[1],  # y-component of the particle's velocity
        particle.mass,  # Mass of the particle
        particle.em_charge,  # Electromagnetic charge of the particle
        particle.element,  # Element type of the particle
        str(particle.color),  # Color of the particle as a string
        particle.em_radius,  # Electromagnetic radius of the particle
        particle.strong_coupling,  # Strong nuclear force coupling constant of the particle
        particle.strong_radius,  # Strong nuclear force radius of the particle
    )
    conn.execute(
        """
        INSERT INTO particles (
            x, y, vx, vy, mass, em_charge, element, color, 
            em_radius, strong_coupling, strong_radius
        ) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        particle_data,
    )  # Insert the particle data into the database
    particle.id = conn.execute("SELECT MAX(id) FROM particles").fetchone()[
        0
    ]  # Retrieve the assigned ID for the particle from the database
    return particle  # Return the created particle instance


def load_particles():
    """
    Load particles from the database.

    Returns:
        list: List of loaded particles.
    """
    particles = []  # Initialize an empty list to store the loaded particles
    for row in conn.execute(
        """
        SELECT 
            id, x, y, vx, vy, mass, em_charge, element, color,
            em_radius, strong_coupling, strong_radius
        FROM particles
        """
    ).fetchall():  # Retrieve all particle data from the database
        particle = Particle(
            row[1], row[2], row[3], row[4], row[7]
        )  # Create a new particle instance with the loaded data
        particle.id, particle.mass = row[0], row[5]  # Set the particle's ID and mass
        particle.em_charge, particle.em_radius = (
            row[6],
            row[9],
        )  # Set the particle's electromagnetic charge and radius
        particle.strong_coupling, particle.strong_radius = (
            row[10],
            row[11],
        )  # Set the particle's strong nuclear force coupling constant and radius
        particle.color = eval(
            row[8]
        )  # Set the particle's color by evaluating the color string
        particles.append(particle)  # Add the loaded particle to the list
    return particles  # Return the list of loaded particles


def save_particles(particles):
    """
    Save particles to the database.

    Args:
        particles (list): List of particles to save.
    """
    conn.execute(
        "DELETE FROM particles"
    )  # Delete all existing particle data from the database
    for particle in particles:
        particle_data = (
            particle.id,  # Unique identifier of the particle
            particle.pos[0],  # x-coordinate of the particle's position
            particle.pos[1],  # y-coordinate of the particle's position
            particle.vel[0],  # x-component of the particle's velocity
            particle.vel[1],  # y-component of the particle's velocity
            particle.mass,  # Mass of the particle
            particle.em_charge,  # Electromagnetic charge of the particle
            particle.element,  # Element type of the particle
            str(particle.color),  # Color of the particle as a string
            particle.em_radius,  # Electromagnetic radius of the particle
            particle.strong_coupling,  # Strong nuclear force coupling constant of the particle
            particle.strong_radius,  # Strong nuclear force radius of the particle
        )
        conn.execute(
            """
            INSERT INTO particles (
                id, x, y, vx, vy, mass, em_charge, element, color,
                em_radius, strong_coupling, strong_radius
            ) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            particle_data,
        )  # Insert the particle data into the database


async def process_particles(particles):
    """
    Process particles asynchronously using a thread pool executor.

    Args:
        particles (list): List of particles to process.
    """
    particles_array = np.array(
        particles, dtype=object
    )  # Convert list to numpy array for efficient processing

    @lru_cache(maxsize=None)  # Cache results to avoid recomputation
    def cached_apply_forces(particle, all_particles):
        return particle.apply_forces(all_particles)

    with ThreadPoolExecutor() as executor:  # Create a thread pool executor
        loop = asyncio.get_event_loop()  # Get the event loop
        forces = await asyncio.gather(
            *[
                loop.run_in_executor(
                    executor, cached_apply_forces, particle, tuple(particles_array)
                )
                for particle in particles_array
            ]
        )  # Run the apply_forces method for each particle concurrently using the thread pool executor

    # Vectorized operations for updating particle velocities and positions
    masses = np.array([particle.mass for particle in particles_array])
    valid_forces = np.array(
        [
            force if force is not None else np.zeros_like(particles_array[0].vel)
            for force in forces
        ]
    )

    velocities = np.array([particle.vel for particle in particles_array])
    positions = np.array([particle.pos for particle in particles_array])

    # Update velocities and positions using numpy for efficient computation
    new_velocities = velocities + valid_forces / masses[:, np.newaxis] * DT
    new_positions = (
        positions
        + new_velocities * DT
        + 0.5 * valid_forces / masses[:, np.newaxis] * DT**2
    )

    # Clip positions to ensure particles stay within boundaries
    new_positions = np.clip(new_positions, 0, [width, height])

    # Update particle objects with new computed values
    for i, particle in enumerate(particles_array):
        particle.vel = new_velocities[i]
        particle.pos = new_positions[i]


def draw_legend(particles, is_expanded=True):
    """
    Draw the legend on the screen, which can be minimized or expanded.

    Args:
        is_expanded (bool): Whether the legend is expanded or collapsed.
    """
    margin = 10
    legend_width = 200
    legend_height = 100 if is_expanded else 20
    legend_rect = pygame.Rect(
        margin, height - legend_height - margin, legend_width, legend_height
    )  # Define the rectangle for the legend dynamically positioned

    # Draw the legend rectangle with a white background
    pygame.draw.rect(screen, (255, 255, 255), legend_rect)

    # Create a font object for the legend text
    font = pygame.font.Font(None, 20)

    # Draw the toggle button for expanding or collapsing the legend
    toggle_text = "-" if is_expanded else "+"
    toggle_button = font.render(toggle_text, True, (0, 0, 0))
    screen.blit(toggle_button, (legend_rect.right - 20, legend_rect.top + 5))

    if is_expanded:
        # Only display element details if the legend is expanded
        visible_elements = [
            e for e in ELEMENTS if any(p.element == e for p in particles)
        ]
        for i, element in enumerate(visible_elements):
            color = ELEMENTS[element]["color"]  # Get the color of the element
            text = font.render(
                element, True, color
            )  # Render the element name with its corresponding color
            screen.blit(
                text, (legend_rect.left + 10, legend_rect.top + 20 + i * 20)
            )  # Draw the element name on the screen


def reset_simulation():
    """
    Reset the simulation by deleting all particles and creating new ones.

    Returns:
        list: List of newly created particles.
    """
    conn.execute(
        "DELETE FROM particles"
    )  # Delete all existing particle data from the database
    particles = [
        create_particle(
            random.randrange(width),  # Random x-coordinate within the simulation width
            random.randrange(
                height
            ),  # Random y-coordinate within the simulation height
            random.uniform(-1, 1),  # Random x-component of velocity between -1 and 1
            random.uniform(-1, 1),  # Random y-component of velocity between -1 and 1
            "hydrogen",  # Only hydrogen particles are created initially
        )
        for _ in range(PARTICLE_COUNT)
    ]  # Create a specified number of particles with random properties
    return particles  # Return the list of newly created particles


def run():
    """
    Run the simulation.
    """
    particles = reset_simulation()  # Reset the simulation and create new particles
    clock = pygame.time.Clock()  # Create a clock object to control the frame rate
    running = True  # Flag to indicate if the simulation is running
    prev_time = pygame.time.get_ticks()  # Get the initial timestamp

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # Quit the simulation if the window is closed
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    particles = (
                        reset_simulation()
                    )  # Reset the simulation if the 'r' key is pressed

        screen.fill((0, 0, 0))  # Fill the screen with black color
        asyncio.run(
            process_particles(particles)
        )  # Process the particles asynchronously

        photon_energies = []  # List to store the energies of emitted photons
        for i, particle in enumerate(particles):
            particle.draw()
            photon_energy = particle.emit_photon()
            if photon_energy > 0:
                photon_energies.append(photon_energy)

            for other in particles[i + 1 :]:
                fusion_photon_energy = particle.fusion(other)
                if fusion_photon_energy > 0:
                    photon_energies.append(fusion_photon_energy)

        particles = [p for p in particles if p.mass > 0]

        draw_legend(particles)
        pygame.display.flip()
        clock.tick(TARGET_FPS)
        save_particles(particles)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run()
