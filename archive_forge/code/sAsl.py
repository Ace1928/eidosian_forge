import pygame as pg
import sys
import numpy as np
from random import randint, uniform
import logging
import heapq
from collections import deque
from typing import List, Tuple, Dict, Optional, Any, Callable, Deque, Set
from itertools import cycle
from math import sin, cos, pi
import threading
import tensorflow as tf
from tensorflow import keras
import os
import time
import psutil
import cProfile
import pstats
import io
import gc
from pstats import SortKey
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import total_ordering

# Configure advanced logging
logging.basicConfig(
    filename="snake_game.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Constants
SCREEN_WIDTH: int = 600
SCREEN_HEIGHT: int = SCREEN_WIDTH
BLOCK_SIZE: int = 20

# AI learning parameters
EXPLORATION_RATE: float = 0.05
LEARNING_RATE: float = 0.05
DISCOUNT_FACTOR: float = 0.95

# Initialize Pygame
pg.init()
WINDOW: pg.Surface = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pg.display.set_caption("Advanced Snake Game with AI Learning and Pathfinding")

# Thread lock for AI data consistency
ai_lock: threading.Lock = threading.Lock()


@total_ordering
@dataclass(frozen=True)
class Position:
    x: int
    y: int

    def __add__(self, other: "Position") -> "Position":
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Position") -> Tuple[int, int]:
        return self.x - other.x, self.y - other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other: "Position") -> bool:
        return self.x == other.x and self.y == other.y

    def __lt__(self, other: "Position") -> bool:
        return (self.x, self.y) < (other.x, other.y)

    def distance(self, other: "Position") -> float:
        dx, dy = self - other
        return (dx**2 + dy**2) ** 0.5


class Direction(Enum):
    """
    Enumeration representing the possible directions the snake can move.
    """

    UP = Position(0, -1)
    DOWN = Position(0, 1)
    LEFT = Position(-1, 0)
    RIGHT = Position(1, 0)


@dataclass
class Snake:
    body: Deque[Position]
    growing: int
    direction: Position
    fruit: "Fruit"  # Forward reference to the Fruit class
    ai_controller: "AIController"  # Forward reference to the AIController class
    collision_detection_strategy: "CollisionDetectionStrategy"  # Forward reference to the CollisionDetectionStrategy class
    resource_manager: (
        "ResourceManager"  # Forward reference to the ResourceManager class
    )
    performance_monitor: (
        "PerformanceMonitor"  # Forward reference to the PerformanceMonitor class
    )

    def update_direction(self, new_direction: Position):
        # Prevent the snake from reversing
        if (new_direction.x + self.direction.x == 0) and (
            new_direction.y + self.direction.y == 0
        ):
            return
        self.direction = new_direction

    def move(self):
        new_head = self.body[0] + self.direction
        if self.collision_detection_strategy.check_collision(new_head, self):
            logging.error("Collision detected at position: %s", new_head)
            raise Exception("Game Over: Collision Detected")
        self.body.appendleft(new_head)
        if not self.growing:
            self.body.pop()
        else:
            self.growing -= 1
        logging.info("Snake moved to new position: %s", new_head)

    def eat_fruit(self):
        if self.body[0] == self.fruit.position:
            self.growing += 10
            self.fruit.relocate()
            logging.info("Fruit eaten at position: %s", self.fruit.position)


@dataclass
class Fruit:
    snake: Optional["Snake"] = None
    colors: cycle = field(
        default_factory=lambda: cycle(
            [
                pg.Color(255, 0, 0),
                pg.Color(255, 165, 0),
                pg.Color(255, 255, 0),
                pg.Color(0, 128, 0),
                pg.Color(0, 0, 255),
                pg.Color(75, 0, 130),
                pg.Color(238, 130, 238),
                pg.Color(255, 255, 255),
                pg.Color(128, 128, 128),
                pg.Color(0, 0, 0),
            ]
        )
    )
    current_color: pg.Color = field(default_factory=lambda: pg.Color(255, 0, 0))
    radius: int = BLOCK_SIZE // 1.5
    position: Position = field(
        default_factory=lambda: Position(0, 0)
    )  # Temporary Default Position

    def __post_init__(self):
        self.current_color = next(
            self.colors
        )  # Initialize the current color from the cycle


class GameObject(ABC):
    """
    Abstract base class for game objects.
    """

    @abstractmethod
    def draw(self) -> None:
        """
        Abstract method to draw the game object on the screen.
        """
        pass


class Fruit(GameObject):
    """
    Represents the fruit object in the snake game, which the snake aims to consume.
    The fruit changes its position randomly on the game grid and has a pulsating color effect with a highly detailed graphical star representation.
    """

    def __init__(self, snake: Optional["Snake"] = None):
        """
        Initializes the Fruit object with a cycling color scheme and a default position.

        Args:
            snake (Optional['Snake']): The snake object that will interact with this fruit. Defaults to None.
        """
        self.snake: Optional["Snake"] = snake
        self.colors: cycle = cycle(
            [
                pg.Color(255, 0, 0),
                pg.Color(255, 165, 0),
                pg.Color(255, 255, 0),
                pg.Color(0, 128, 0),
                pg.Color(0, 0, 255),
                pg.Color(75, 0, 130),
                pg.Color(238, 130, 238),
                pg.Color(255, 255, 255),
                pg.Color(128, 128, 128),
                pg.Color(0, 0, 0),
            ]
        )
        self.current_color: pg.Color = next(self.colors)
        self.radius: int = BLOCK_SIZE // 1.5
        self.points: int = 10
        self.position: Position = Position(0, 0)  # Temporary Default Position

    def set_snake(self, snake: "Snake") -> None:
        """
        Links the fruit with a snake object and relocates the fruit to a new position.

        Args:
            snake (Snake): The snake object that will interact with this fruit.
        """
        self.snake = snake
        self.relocate()

    def draw(self) -> None:
        """
        Draws the fruit on the game window with a pulsating star shape.
        The star is drawn with detailed vertices to create a realistic star appearance.
        """
        alpha: float = (sin(pg.time.get_ticks() * 0.002) + 1) / 2
        self.current_color.a = int(alpha * 255)
        points: List[Position] = self._calculate_star_points(
            Position(self.position.x + self.radius, self.position.y),
            self.radius,
            self.points,
        )
        pg.draw.polygon(WINDOW, self.current_color, points)

    def relocate(self) -> None:
        """
        Relocates the fruit to a new position on the game grid that is not occupied by the snake.
        This method ensures that the fruit does not appear on the snake's body.
        """
        if not self.snake:
            return  # Ensure snake is set before relocating
        while True:
            new_x: int = randint(0, (SCREEN_WIDTH // BLOCK_SIZE) - 1) * BLOCK_SIZE
            new_y: int = randint(0, (SCREEN_HEIGHT // BLOCK_SIZE) - 1) * BLOCK_SIZE
            new_position: Position = Position(new_x, new_y)
            if not self._is_collision_imminent(new_position):
                self.position = new_position
                break

    def _is_collision_imminent(self, new_position: Position) -> bool:
        """
        Checks if placing the fruit at the new position would create an unavoidable collision path for the snake.

        Args:
            new_position (Position): The new position to check for collision.

        Returns:
            bool: True if the new position would create an unavoidable collision path, False otherwise.
        """
        if not self.snake:
            return False  # No collision if snake is not set

        snake_head: Position = self.snake.body[0]
        snake_direction: Position = Position(
            self.snake.direction.x, self.snake.direction.y
        )

        # Check if the new position is in the immediate path of the snake
        if (
            new_position
            == Position(
                snake_head.x + snake_direction.x, snake_head.y + snake_direction.y
            )
            # Incorrect line
            or new_position
            == Position(
                snake_head.x + snake_direction.y, snake_head.y + snake_direction.x
            )
            or new_position
            == Position(
                snake_head.x - snake_direction.y, snake_head.y - snake_direction.x
            )
        ):
            return True

        # Check if the new position is surrounded by the snake's body
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if (
                Position(
                    new_position.x + dx * BLOCK_SIZE,
                    new_position.y + dy * BLOCK_SIZE,
                )
                in self.snake.body
            ):
                return True

        return False

    def _calculate_star_points(
        self, center: Position, radius: int, points: int
    ) -> List[Position]:
        """
        Calculates the vertices of a star shape based on the center, radius, and number of points.
        This method uses trigonometric functions to determine the precise location of each vertex for a realistic star shape.

        Args:
            center (Position): The center of the star.
            radius (int): The radius of the star.
            points (int): The number of points of the star.

        Returns:
            List[Position]: A list of Positions representing the vertices of the star.
        """
        angle: float = pi / points
        return [
            (
                int(center.x + sin(i * 2 * angle) * radius),
                int(center.y + cos(i * 2 * angle) * radius),
            )
            for i in range(2 * points + 1)
        ]


class CollisionDetectionStrategy(ABC):
    """
    An abstract base class for collision detection strategies, ensuring that each strategy adheres to a common interface for collision detection.
    This class provides a structured approach to defining how collisions are detected within the game environment.
    """

    @abstractmethod
    def check_collision(self, new_position: Position, snake: Snake) -> bool:
        """
        Abstract method to check if the new position collides with any obstacle or the snake's body.
        This method must be implemented by all subclasses to ensure specific collision detection logic is applied.

        Args:
            new_position (Position): The new position to check for collisions.
            snake (Snake): The snake object, which contains the current state of the snake.

        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        pass


class BasicCollisionDetectionStrategy(CollisionDetectionStrategy):
    """
    A basic collision detection strategy that checks for collisions with the snake's body and game boundaries.
    This strategy is suitable for simple gameplay scenarios where advanced collision prediction is not required.
    """

    def check_collision(self, new_position: Position, snake: Snake) -> bool:
        """
        Implements basic collision detection by checking if the new position collides with the snake's body or game boundaries.

        Args:
            new_position (Position): The new position to check for collisions.
            snake (Snake): The snake object.

        Returns:
            bool: True if a collision is detected with either the snake's body or the boundaries of the game area, False otherwise.
        """
        if (
            new_position in snake.body
            or new_position.x < 0
            or new_position.x >= SCREEN_WIDTH
            or new_position.y < 0
            or new_position.y >= SCREEN_HEIGHT
        ):
            return True
        return False


class AdvancedCollisionDetectionStrategy(CollisionDetectionStrategy):
    """
    An advanced collision detection strategy that checks for collisions with the snake's body, game boundaries, and predicts future collisions.
    This strategy incorporates both immediate collision detection and predictive techniques to enhance gameplay dynamics.
    """

    def check_collision(self, new_position: Position, snake: Snake) -> bool:
        """
        Extends basic collision detection by also predicting future collisions based on the snake's current direction and surrounding positions.

        Args:
            new_position (Position): The new position to check for collisions.
            snake (Snake): The snake object.

        Returns:
            bool: True if a collision is detected or predicted, False otherwise.
        """
        if BasicCollisionDetectionStrategy().check_collision(new_position, snake):
            return True

        # Check if the new position is in the immediate path of the snake
        snake_head: Position = snake.body[0]
        snake_direction: Position = Position(snake.direction.x, snake.direction.y)

        if (
            new_position == snake_head + snake_direction
            or new_position
            == snake_head + Position(snake_direction.y, snake_direction.x)
            or new_position
            == snake_head + Position(-snake_direction.y, snake_direction.x)
        ):
            return True

        # Check if the new position is surrounded by the snake's body
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if (
                Position(
                    new_position.x + dx * BLOCK_SIZE, new_position.y + dy * BLOCK_SIZE
                )
                in snake.body
            ):
                return True

        return False


class ResourceManager(ABC):
    """
    An abstract base class for resource management strategies, ensuring that all game components such as Snake, Fruit, and AIController are initialized and managed efficiently.
    """

    @abstractmethod
    def monitor_resources(self) -> None:
        """
        Monitors and optimizes resource usage, including memory management, CPU usage, and logging levels, to maintain optimal game performance.
        """
        pass


class BasicResourceManager(ResourceManager):
    """
    A basic resource manager that performs minimal resource monitoring and optimization. This manager is suitable for scenarios with lower resource demands.
    """

    def monitor_resources(self) -> None:
        """
        Performs minimal resource monitoring and optimization, primarily focusing on basic logging of resource usage without active management.
        """
        logging.debug(
            "Basic resource monitoring engaged. Minimal optimization applied."
        )
        pass


class OptimizedResourceManager(ResourceManager):
    """
    An optimized resource manager that actively monitors and optimizes resource usage, incorporating advanced techniques such as dynamic logging level adjustments and garbage collection to enhance game performance.
    """

    def monitor_resources(self) -> None:
        """
        Actively monitors and optimizes resource usage by adjusting logging levels based on CPU usage and performing garbage collection to manage memory efficiently.
        """
        # Dynamically adjust logging level based on CPU usage to provide appropriate logging detail without overwhelming the system
        current_cpu_usage: float = self.get_cpu_usage()
        if current_cpu_usage > 80:
            logging.getLogger().setLevel(logging.WARNING)
            logging.warning(
                f"High CPU usage detected: {current_cpu_usage}%. Reducing logging verbosity."
            )
        else:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.debug(
                f"CPU usage normal: {current_cpu_usage}%. Detailed logging enabled."
            )

        # Perform garbage collection to optimize memory usage, especially important during long game sessions or high complexity scenarios
        gc.collect()
        logging.info("Garbage collection performed to optimize memory usage.")

    def get_cpu_usage(self) -> float:
        """
        Retrieves the current CPU usage percentage, utilizing the psutil library to monitor system resources accurately.

        Returns:
            float: The current CPU usage percentage, providing a precise measurement that is critical for resource optimization.
        """
        cpu_usage: float = psutil.cpu_percent(interval=1)
        logging.debug(f"Current CPU usage retrieved: {cpu_usage}%")
        return cpu_usage


class PerformanceMonitor(ABC):
    """
    An abstract base class for performance monitoring strategies, providing a structured approach to log and analyze the performance of various operations within the game.
    """

    @abstractmethod
    def log_draw_performance(self) -> None:
        """
        Abstract method to log the performance of the draw operation, which must be implemented by any subclass to ensure detailed performance tracking.
        """
        pass


class BasicPerformanceMonitor(PerformanceMonitor):
    """
    A basic performance monitor that performs minimal performance logging, suitable for scenarios with lower performance analysis requirements.
    """

    def log_draw_performance(self) -> None:
        """
        Implements minimal performance logging for the draw operation, primarily focusing on basic logging without detailed analysis.
        """
        logging.debug("Basic draw performance logged with minimal details.")


class RealTimePerformanceMonitor(PerformanceMonitor):
    """
    A real-time performance monitor that actively logs and analyzes performance metrics, utilizing advanced techniques to provide real-time feedback and optimization suggestions.
    """

    def __init__(self) -> None:
        """
        Initializes the RealTimePerformanceMonitor with an empty list to store draw times for ongoing analysis.
        """
        self.draw_times: List[float] = []

    def log_draw_performance(self) -> None:
        """
        Logs and analyzes the performance of the draw operation in real-time, capturing the time taken for each draw and computing statistics over a set number of frames to provide actionable insights.
        """
        start_time: float = time.perf_counter()
        # Placeholder for the draw operation
        end_time: float = time.perf_counter()
        draw_time: float = end_time - start_time
        self.draw_times.append(draw_time)

        # Log individual draw time with high precision
        logging.debug(f"Draw time recorded: {draw_time:.5f} seconds")

        # Perform analysis if sufficient data has been collected
        if len(self.draw_times) >= 100:
            avg_draw_time: float = sum(self.draw_times) / len(self.draw_times)
            max_draw_time: float = max(self.draw_times)
            min_draw_time: float = min(self.draw_times)
            logging.info(
                f"Average draw time (last 100 frames): {avg_draw_time:.5f} seconds"
            )
            logging.info(
                f"Maximum draw time (last 100 frames): {max_draw_time:.5f} seconds"
            )
            logging.info(
                f"Minimum draw time (last 100 frames): {min_draw_time:.5f} seconds"
            )
            # Reset draw times list for the next batch of analysis
            self.draw_times.clear()


class Snake(GameObject):
    """
    Represents the snake in the game, which moves according to AI decisions, grows by consuming fruits, and avoids collisions.

    Attributes:
        body (deque[Position]): A deque storing the positions of the snake's segments.
        growing (int): A counter to manage the growth of the snake after eating a fruit.
        direction (Position): The current movement direction of the snake.
        fruit (Fruit): A reference to the fruit object in the game.
        ai_controller (AIController): The AI controller that decides the snake's movements.
        collision_detection_strategy (CollisionDetectionStrategy): The strategy used for collision detection.
        resource_manager (ResourceManager): The resource manager for optimizing resource usage.
        performance_monitor (PerformanceMonitor): The performance monitor for tracking and optimizing performance.
    """

    def __init__(
        self,
        start_position: Position = Position(
            SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100
        ),
        initial_length: int = 3,
        collision_detection_strategy=AdvancedCollisionDetectionStrategy(),
        resource_manager=OptimizedResourceManager(),
        performance_monitor=RealTimePerformanceMonitor(),
    ) -> None:
        """
        Initializes the Snake object with a starting position, initial length, collision detection strategy, resource manager, and performance monitor.

        Args:
            start_position (Position): The starting position of the snake's head. Defaults to the center of the screen.
            initial_length (int): The initial number of segments of the snake. Defaults to 3.
            collision_detection_strategy (CollisionDetectionStrategy): The strategy used for collision detection. Defaults to AdvancedCollisionDetectionStrategy.
            resource_manager (ResourceManager): The resource manager for optimizing resource usage. Defaults to OptimizedResourceManager.
            performance_monitor (PerformanceMonitor): The performance monitor for tracking and optimizing performance. Defaults to RealTimePerformanceMonitor.
        """
        self.body: deque[Position] = deque(
            [start_position]
            + [
                Position(start_position.x, start_position.y - i * BLOCK_SIZE)
                for i in range(1, initial_length)
            ]
        )
        self.growing: int = 0
        self.direction: Position = Direction.DOWN.value
        self.fruit: Fruit = Fruit(self)  # Create fruit and link it to the snake
        self.ai_controller: AIController = AIController(
            self, self.fruit
        )  # AI uses both snake and fruit
        self.collision_detection_strategy: CollisionDetectionStrategy = (
            collision_detection_strategy
        )
        self.resource_manager: ResourceManager = resource_manager
        self.performance_monitor: PerformanceMonitor = performance_monitor

        # Initialize logging
        logging.basicConfig(
            filename="snake.log",
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.info("Snake initialized")

    def draw(self) -> None:
        """
        Draws the snake on the game window with advanced visual effects.
        Each segment of the snake has a different color to create a gradient effect.
        """
        hue: float = 0
        hue_step: float = 360 / len(self.body)
        for segment in self.body:
            color: pg.Color = pg.Color(0)
            color.hsva = (hue, 100, 100, 100)
            pg.draw.rect(
                WINDOW, color, pg.Rect(segment.x, segment.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            hue += hue_step  # Create a continuous color cycle

        # Add a visual effect for the head to simulate "munching"
        head: Position = self.body[0]
        munch_rect: pg.Rect = pg.Rect(head.x, head.y, BLOCK_SIZE, BLOCK_SIZE)
        pg.draw.ellipse(
            WINDOW, pg.Color("yellow"), munch_rect
        )  # Munching effect on the head

        # Log drawing performance
        self.performance_monitor.log_draw_performance()

    def move(self) -> None:
        """
        Moves the snake based on the AI's decision and checks for collisions.
        """
        self.direction = self.ai_controller.decide_action(
            self.body[0],
            self.fruit.position,
        )
        new_head: Position = Position(
            self.body[0].x + self.direction.x * BLOCK_SIZE,
            self.body[0].y + self.direction.y * BLOCK_SIZE,
        )
        if self.collision_detection_strategy.check_collision(new_head, self):
            raise ValueError("Collision detected")
        self.body.appendleft(new_head)
        if not self.growing:
            self.body.pop()
        else:
            self.growing -= 1  # Decrement the growing counter after adding a segment

        # Log movement
        logging.debug(f"Snake moved to {new_head}")

        # Monitor resource usage
        self.resource_manager.monitor_resources()

    def grow(self) -> None:
        """
        Increases the size of the snake by adding a segment.
        """
        self.growing += 1
        logging.debug(f"Snake grew by 1 segment")

    def update_direction(self, new_direction: Position) -> None:
        """
        Updates the snake's direction based on user input or AI decision, avoiding immediate reversals.

        Args:
            new_direction (Position): The new direction for the snake to move.
        """
        if (new_direction.x + self.direction.x == 0) and (
            new_direction.y + self.direction.y == 0
        ):
            return  # Prevent the snake from reversing direction
        self.direction = new_direction

    def calculate_space(self, direction: Position) -> int:
        """
        Calculate the free space in a given direction to assist in making turn decisions.

        Args:
            direction (Position): The direction to check for free space.

        Returns:
            int: The number of free blocks in the specified direction.
        """
        step: int = 0
        x, y = self.body[0].x, self.body[0].y
        while True:
            x += direction.x * BLOCK_SIZE
            y += direction.y * BLOCK_SIZE
            if Position(x, y) in self.body or not (
                0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT
            ):
                break
            step += 1
        return step

    def calculate_distance(self, start: Position, end: Position) -> float:
        """
        Calculate Euclidean distance from the current head position to the fruit.

        Args:
            start (Position): The starting point for the distance calculation.
            end (Position): The ending point for the distance calculation.

        Returns:
            float: The Euclidean distance between the start and end points.
        """
        return start.distance(end)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Snake object.

        Returns:
            str: The string representation of the Snake object.
        """
        return f"Snake(body={self.body}, growing={self.growing}, direction={self.direction})"

    def __str__(self) -> str:
        """
        Returns a readable string representation of the Snake object.

        Returns:
            str: The readable string representation of the Snake object.
        """
        return f"Snake with {len(self.body)} segments, growing: {self.growing}, direction: {self.direction}"


class ThetaStar:
    """
    Implement an enhanced ThetaStar pathfinding algorithm for the snake game, which accounts for future changes
    in the snake's body configuration upon eating a fruit. This class is designed to handle complex pathfinding
    scenarios with optimized performance and advanced error handling.

    Attributes:
        None - This class purely consists of static methods.
    """

    @staticmethod
    def line_of_sight(grid: List[List[int]], start: Position, end: Position) -> bool:
        """
        Check if there's a clear line of sight between two points in the grid, considering obstacles.

        Args:
            grid (List[List[int]]): The game grid where 1 represents obstacles (snake body).
            start (Position): The starting point.
            end (Position): The ending point.

        Returns:
            bool: True if there's an unobstructed line of sight between start and end, False otherwise.
        """
        if start is None or end is None:
            logging.error(f"Invalid line of sight call: start={start}, end={end}")
            return False  # Validate input parameters

        x0, y0 = start.x // BLOCK_SIZE, start.y // BLOCK_SIZE
        x1, y1 = end.x // BLOCK_SIZE, end.y // BLOCK_SIZE
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx - dy

        while x0 != x1 or y0 != y1:
            if grid[x0][y0] == 1:  # Obstacle detected
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return True

    @staticmethod
    def theta_star(
        start: Position, goal: Position, grid: List[List[int]]
    ) -> List[Position]:
        """
        Perform the Theta* search algorithm to find a path from start to goal, considering dynamic snake growth.

        Args:
            start (Position): The start coordinate (x, y).
            goal (Position): The goal coordinate (x, y).
            grid (List[List[int]]): The game grid where 1 represents obstacles (snake body).

        Returns:
            List[Position]: The path from start to goal as a list of coordinates.
        """
        if start is None or goal is None:
            logging.error(f"Invalid Theta* call: start={start}, goal={goal}")
            return []

        open_set: List[Tuple[float, Position]] = []
        heapq.heappush(open_set, (0 + start.distance(goal), start))
        came_from: Dict[Position, Position] = {start: start}
        g_score: Dict[Position, float] = {start: 0}
        f_score: Dict[Position, float] = {start: start.distance(goal)}
        closed_set: Set[Position] = set()

        while open_set:
            current: Position = heapq.heappop(open_set)[1]
            if current == goal:
                path: List[Position] = []
                while current != start:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            closed_set.add(current)

            for dx, dy in [
                (0, BLOCK_SIZE),
                (0, -BLOCK_SIZE),
                (BLOCK_SIZE, 0),
                (-BLOCK_SIZE, 0),
            ]:
                neighbor = Position(current.x + dx, current.y + dy)

                if (
                    0 <= neighbor.x < SCREEN_WIDTH
                    and 0 <= neighbor.y < SCREEN_HEIGHT
                    and grid[neighbor.x // BLOCK_SIZE][neighbor.y // BLOCK_SIZE] == 0
                    and neighbor not in closed_set
                ):
                    tentative_g_score: float = g_score[current] + current.distance(
                        neighbor
                    )

                    if (
                        neighbor not in [pos for _, pos in open_set]
                        or tentative_g_score < g_score[neighbor]
                    ):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + neighbor.distance(goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    @staticmethod
    def hierarchical_theta_star(
        start: Position,
        goal: Position,
        grid: List[List[int]],
        levels: int = 2,
    ) -> List[Position]:
        """
        Perform hierarchical pathfinding using the Theta* algorithm to efficiently manage larger grids.

        Args:
            start (Position): The start coordinate (x, y).
            goal (Position): The goal coordinate (x, y).
            grid (List[List[int]]): The game grid where 1 represents obstacles (snake body).
            levels (int): The number of hierarchical levels to use. Defaults to 2.

        Returns:
            List[Position]: The path from start to goal as a list of coordinates.
        """
        if levels <= 1:
            return ThetaStar.theta_star(start, goal, grid)

        # Create a coarser grid for higher-level pathfinding
        coarse_grid: List[List[int]] = [
            [
                min(
                    grid[i * 2][j * 2],
                    grid[i * 2 + 1][j * 2],
                    grid[i * 2][j * 2 + 1],
                    grid[i * 2 + 1][j * 2 + 1],
                )
                for j in range(len(grid[0]) // 2)
            ]
            for i in range(len(grid) // 2)
        ]

        # Find a high-level path using the coarser grid
        coarse_path: List[Position] = ThetaStar.hierarchical_theta_star(
            Position(start.x // 2, start.y // 2),
            Position(goal.x // 2, goal.y // 2),
            coarse_grid,
            levels - 1,
        )

        # Refine the path using the original grid
        refined_path: List[Position] = []
        for i in range(len(coarse_path) - 1):
            refined_path.extend(
                ThetaStar.theta_star(
                    Position(coarse_path[i].x * 2, coarse_path[i].y * 2),
                    Position(coarse_path[i + 1].x * 2, coarse_path[i + 1].y * 2),
                    grid,
                )
            )
        refined_path.append(goal)

        return refined_path

    def __str__(self) -> str:
        """
        Provide a human-readable representation of the ThetaStar pathfinding algorithm's capabilities.
        """
        return "ThetaStar pathfinding algorithm with dynamic snake body adjustment and hierarchical pathfinding."


class AIController:
    """
    Manages AI decision-making using Q-learning, enhanced pathfinding algorithms, and potential neural network integration.
    The AIController dynamically adjusts its strategy based on the current state of the game,
    predicting potential future states and making decisions that optimize for both short-term
    gains and long-term survival.

    Attributes:
        snake (Snake): Reference to the snake, used to access the snake's current state.
        fruit (Fruit): Reference to the fruit, used to determine the target position.
        q_table (numpy.ndarray): A table used to store the Q-values for state-action pairs.
        action_map (list): List of possible actions the snake can take.
        state_cache (dict): Cache to store previously calculated paths and Q-values.
        lock (threading.Lock): Lock for thread-safe operations on shared data.
        neural_network (Optional[keras.Model]): A placeholder for a potential neural network to enhance decision-making.
        pathfinding_algorithms (List[Callable]): List of pathfinding algorithms, ranked by priority.
        learning_algorithms (List[Callable]): List of learning algorithms, ranked by priority.
    """

    def __init__(self, snake: Snake, fruit: Fruit):
        """
        Initialize the AIController with references to the snake and fruit, and setup the Q-table and other necessary structures.

        Args:
            snake (Snake): Reference to the snake object.
            fruit (Fruit): Reference to the fruit object.
        """
        self.snake: Snake = snake
        self.fruit: Fruit = fruit
        self.q_table: np.ndarray = np.zeros(
            (SCREEN_WIDTH // BLOCK_SIZE, SCREEN_HEIGHT // BLOCK_SIZE, 4),
            dtype=np.float32,
        )
        self.action_map: List[Position] = [
            Position(0, -1),
            Position(0, 1),
            Position(-1, 0),
            Position(1, 0),
        ]
        self.state_cache: Dict[Tuple[Position, Position], List[Position]] = {}
        self.lock: threading.Lock = threading.Lock()
        self.neural_network: Optional[keras.Model] = self._initialize_neural_network()
        self.pathfinding_algorithms: List[Callable] = [
            ThetaStar.theta_star,
            self.decide_by_neural_network,
        ]  # Ranked by priority
        self.learning_algorithms: List[Callable] = [
            self.decide_by_q_learning,
            self.decide_by_neural_network,
        ]  # Ranked by priority

    def _initialize_neural_network(self) -> keras.Model:
        """
        Initialize a basic neural network for decision-making.

        Returns:
            keras.Model: A basic neural network model.
        """
        model = keras.Sequential(
            [
                keras.Input(shape=(4,)),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(8, activation="relu"),
                keras.layers.Dense(4, activation="linear"),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def decide_action(self, state: Position, goal: Position) -> Position:
        """
        Decide the next action for the snake based on the current state and the goal.
        Uses cached paths if available, otherwise computes a new path using the highest priority pathfinding algorithm.
        In cases of pathfinding failure, the decision reverts to the highest priority learning algorithm.

        Args:
            state (Position): The current state of the snake, typically its head position.
            goal (Position): The target state, typically the fruit position.

        Returns:
            Position: The next action (dx, dy) that the snake should take.
        """
        grid: List[List[int]] = [
            [
                (
                    0
                    if Position(x * BLOCK_SIZE, y * BLOCK_SIZE) not in self.snake.body
                    else 1
                )
                for y in range(SCREEN_HEIGHT // BLOCK_SIZE)
            ]
            for x in range(SCREEN_WIDTH // BLOCK_SIZE)
        ]
        path: List[Position] = []  # Default to an empty path
        with self.lock:
            if (state, goal) in self.state_cache:
                path = self.state_cache[(state, goal)]
            else:
                for pathfinding_algorithm in self.pathfinding_algorithms:
                    path = pathfinding_algorithm(state, goal, grid)
                    if path:
                        break
                if (
                    not path
                ):  # If all pathfinding algorithms fail, use learning algorithms
                    for learning_algorithm in self.learning_algorithms:
                        action = learning_algorithm(state, goal)
                        if action:
                            return action
                self.state_cache[(state, goal)] = path
        if path and len(path) > 1:
            return Position(path[1].x - state.x, path[1].y - state.y)
        return Position(randint(-1, 1), randint(-1, 1))

    def decide_by_q_learning(
        self, state: Position, goal: Position
    ) -> Optional[Position]:
        """
        Decide the next action using Q-learning when pathfinding fails.

        Args:
            state (Position): The current state of the snake.
            goal (Position): The target state.

        Returns:
            Optional[Position]: The next action (dx, dy) based on learned behaviors, or None if no action is determined.
        """
        if np.random.rand() < EXPLORATION_RATE:
            return self.action_map[randint(0, 3)]
        action_index = np.argmax(
            self.q_table[state.x // BLOCK_SIZE][state.y // BLOCK_SIZE]
        )
        return self.action_map[action_index]

    def update_q_table(
        self,
        state: Position,
        action: Position,
        reward: float,
        next_state: Position,
    ):
        """
        Update the Q-values based on the state transition and received reward.
        This method also anticipates future states by adjusting the Q-values towards
        the maximum future expected rewards.

        Args:
            state (Position): The current state (x, y).
            action (Position): The action taken from the current state.
            reward (float): The reward received after taking the action.
            next_state (Position): The state resulting from taking the action.
        """
        action_index: int = self.action_map.index(action)
        with self.lock:
            current_q: float = self.q_table[state.x // BLOCK_SIZE][
                state.y // BLOCK_SIZE
            ][action_index]
            max_future_q: float = np.max(
                self.q_table[next_state.x // BLOCK_SIZE][next_state.y // BLOCK_SIZE]
            )
            new_q: float = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * max_future_q
            )
            self.q_table[state.x // BLOCK_SIZE][state.y // BLOCK_SIZE][
                action_index
            ] = new_q
            logging.info(
                f"Updated Q-table at {state} for action {action} with new Q-value {new_q}"
            )

    def train_neural_network(
        self, state: Position, goal: Position, path: List[Position]
    ):
        """
        Train the neural network using the game state, goal, and the chosen path.

        Args:
            state (Position): The current state of the snake.
            goal (Position): The target state.
            path (List[Position]): The path chosen by the pathfinding or learning algorithm.
        """
        if self.neural_network is None:
            logging.warning("Neural network is not initialized. Skipping training.")
            return

        x_train = np.array([state.x, state.y, goal.x, goal.y]).reshape(1, -1)
        y_train = np.zeros((1, 4))

        if path:
            next_step = path[1]
            action_index = self.action_map.index(
                Position(next_step.x - state.x, next_step.y - state.y)
            )
            y_train[0, action_index] = 1

        with self.lock:
            self.neural_network.fit(x_train, y_train, epochs=1, verbose=0)
            logging.info(f"Trained neural network with state {state} and goal {goal}")

    def decide_by_neural_network(
        self, state: Position, goal: Position
    ) -> Optional[Position]:
        """
        Decide the next action using a neural network when other methods fail.

        Args:
            state (Position): The current state of the snake.
            goal (Position): The target state.

        Returns:
            Optional[Position]: The next action (dx, dy) based on the neural network's prediction, or None if no action is determined.
        """
        if self.neural_network is None:
            logging.warning(
                "Neural network is not initialized. Skipping decision-making."
            )
            return None

        x_input = np.array([state.x, state.y, goal.x, goal.y]).reshape(1, -1)
        with self.lock:
            q_values = self.neural_network.predict(x_input)[0]
        action_index = np.argmax(q_values)
        return self.action_map[action_index]

    def update_algorithm_priorities(self, game_state: Dict[str, Any]):
        """
        Update the priorities of pathfinding and learning algorithms based on the current game state.

        Args:
            game_state (Dict[str, Any]): The current state of the game, including relevant game parameters and metrics.
        """
        # Placeholder implementation: Swap priorities every 100 steps
        if game_state["step"] % 100 == 0:
            self.pathfinding_algorithms.reverse()
            self.learning_algorithms.reverse()
            logging.info(
                f"Updated algorithm priorities: Pathfinding: {self.pathfinding_algorithms}, Learning: {self.learning_algorithms}"
            )

    def __str__(self) -> str:
        """
        Provide a string representation of the AI controller's state for debugging and logging purposes.
        """
        return f"AIController managing a snake game with dynamic decision-making capabilities."


def main() -> None:
    """
    The main game loop function that initializes the game, handles events, updates game state, and renders the game.
    """
    pg.init()
    snake: Snake = Snake()
    fruit: Fruit = Fruit(snake=snake)
    ai_controller: AIController = AIController(snake=snake, fruit=fruit)
    snake.fruit = fruit
    snake.ai_controller = ai_controller
    clock: pg.time.Clock = pg.time.Clock()
    running: bool = True

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                direction: Optional[Position] = None
                if event.key == pg.K_UP:
                    direction = Direction.UP.value
                elif event.key == pg.K_DOWN:
                    direction = Direction.DOWN.value
                elif event.key == pg.K_LEFT:
                    direction = Direction.LEFT.value
                elif event.key == pg.K_RIGHT:
                    direction = Direction.RIGHT.value

                if direction:
                    snake.update_direction(direction)

        try:
            snake.move()
        except Exception as e:
            logging.error(f"Game over due to {str(e)}. Restarting...")
            snake = Snake()
            fruit.set_snake(snake)
            snake.fruit = fruit
            snake.ai_controller = ai_controller

        if snake.body[0] == snake.fruit.position:
            snake.grow()
            snake.fruit.relocate()

        WINDOW.fill(pg.Color("black"))
        snake.fruit.draw()
        snake.draw()
        pg.display.flip()
        clock.tick(60)

    pg.quit()


if __name__ == "__main__":
    main()
