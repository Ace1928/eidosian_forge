import numpy as np
import numba
from numba import njit
import pygame
from typing import List

# Constants
G = 6.67430e-11  # Gravitational constant
DT = 0.01  # Time step for simulation
CENTRAL_MASS = 1.989e30  # Mass of a central body, e.g., the Sun
CENTRAL_POSITION = np.array(
    [400, 300], dtype=np.float64
)  # Central position for gravity pull
COLLISION_THRESHOLD = 10  # Threshold distance for collision detection


class Body:
    """Represents a celestial body within the simulation."""

    def __init__(self, mass: float, position: np.ndarray, velocity: np.ndarray):
        """
        Initialize a new celestial body.

        Args:
            mass (float): The mass of the body.
            position (np.ndarray): The 2D position of the body, expected to be a float array.
            velocity (np.ndarray): The 2D velocity of the body, expected to be a float array.
        """
        self.mass = mass
        self.position = position.astype(np.float64)  # Ensure the position is float64
        self.velocity = velocity.astype(np.float64)  # Ensure the velocity is float64

    def __repr__(self) -> str:
        return f"Body(mass={self.mass}, position={self.position}, velocity={self.velocity})"


class Simulation:
    """Handles the orbital simulation of bodies under gravity."""

    def __init__(self):
        """Initialize the simulation with an empty list of bodies."""
        self.bodies: List[Body] = []

    def add_body(self, body: Body) -> None:
        """Add a new body to the simulation."""
        self.bodies.append(body)

    def remove_body(self, index: int) -> None:
        """Remove a body from the simulation by its index."""
        if 0 <= index < len(self.bodies):
            del self.bodies[index]

    @staticmethod
    @njit(fastmath=True)
    def compute_gravitational_force(
        positions: np.ndarray, masses: np.ndarray
    ) -> np.ndarray:
        """Compute the gravitational forces between all bodies and a central massive body."""
        num_bodies = positions.shape[0]
        forces = np.zeros((num_bodies, 2), dtype=np.float64)

        for i in range(num_bodies):
            # Central gravity pull
            delta_pos_central = CENTRAL_POSITION - positions[i]
            distance_central = np.linalg.norm(delta_pos_central)
            if distance_central > 1e-5:
                force_magnitude_central = (
                    G * CENTRAL_MASS * masses[i] / distance_central**2
                )
                forces[i] += (
                    force_magnitude_central * delta_pos_central / distance_central
                )

            # Gravitational forces between bodies
            for j in range(i + 1, num_bodies):
                delta_pos = positions[j] - positions[i]
                distance = np.linalg.norm(delta_pos)
                if distance > 1e-5:
                    force_magnitude = G * masses[i] * masses[j] / distance**2
                    force = force_magnitude * delta_pos / distance
                    forces[i] += force
                    forces[j] -= force

        return forces

    @staticmethod
    @njit(fastmath=True)
    def handle_collisions(
        positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray
    ) -> np.ndarray:
        """Handle elastic collisions between bodies."""
        num_bodies = positions.shape[0]
        collision_indices = []

        for i in range(num_bodies):
            for j in range(i + 1, num_bodies):
                delta_pos = positions[j] - positions[i]
                distance = np.linalg.norm(delta_pos)
                if distance < COLLISION_THRESHOLD:
                    collision_indices.append((i, j))

        for i, j in collision_indices:
            # Elastic collision
            total_mass = masses[i] + masses[j]
            velocities[i] = (
                masses[i] * velocities[i] + masses[j] * velocities[j]
            ) / total_mass
            velocities[j] = velocities[i]

        return velocities

    def update(self) -> None:
        """Update the positions and velocities of all bodies."""
        if len(self.bodies) < 2:
            return  # Not enough bodies to simulate

        positions = np.array([body.position for body in self.bodies], dtype=np.float64)
        velocities = np.array([body.velocity for body in self.bodies], dtype=np.float64)
        masses = np.array([body.mass for body in self.bodies], dtype=np.float64)

        forces = self.compute_gravitational_force(positions, masses)
        velocities += forces / masses[:, np.newaxis] * DT
        positions += velocities * DT

        velocities = self.handle_collisions(positions, velocities, masses)

        for i, body in enumerate(self.bodies):
            body.position = positions[i]
            body.velocity = velocities[i]


def run_simulation() -> None:
    """Main function to run the orbital simulation with graphical interface."""
    pygame.init()
    screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
    pygame.display.set_caption("Orbital Simulation")
    clock = pygame.time.Clock()
    paused = False

    # Initialize simulation
    sim = Simulation()
    sim.add_body(
        Body(
            5.972e24,
            np.array([400, 300], dtype=np.float64),
            np.array([0, 0], dtype=np.float64),
        )
    )
    sim.add_body(
        Body(
            7.348e22,
            np.array([800, 300], dtype=np.float64),
            np.array([0, -1.022], dtype=np.float64),
        )
    )

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mass = np.random.uniform(1e22, 1e25)
                    position = np.array(pygame.mouse.get_pos(), dtype=np.float64)
                    velocity = np.array([0, 0], dtype=np.float64)
                    sim.add_body(Body(mass, position, velocity))
                elif event.button == 3:  # Right click
                    mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.float64)
                    distances = np.array(
                        [
                            np.linalg.norm(body.position - mouse_pos)
                            for body in sim.bodies
                        ]
                    )
                    if distances.size > 0:
                        sim.remove_body(np.argmin(distances))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            sim.update()

        screen.fill((0, 0, 0))
        pygame.draw.circle(screen, (255, 255, 0), CENTRAL_POSITION.astype(int), 10)
        for body in sim.bodies:
            pygame.draw.circle(screen, (255, 255, 255), body.position.astype(int), 5)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    run_simulation()
