import pygame as pg
import sys
import numpy as np
import random
from random import randint, uniform, choice
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

# Object Constants
FRUIT = "Fruit"
SNAKE = "Snake"
GRID = "Grid"

# Initialize Pygame
pg.init()
WINDOW: pg.Surface = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pg.display.set_caption("Advanced Snake Game with AI Learning and Pathfinding")

# Thread lock for AI data consistency
ai_lock: threading.Lock = threading.Lock()


@total_ordering
@dataclass(frozen=True)
class Position:
    """
    Represents a position on the game grid.

    Attributes:
        x (int): The x-coordinate of the position.
        y (int): The y-coordinate of the position.
    """

    x: int
    y: int

    def __add__(self, other: "Position") -> "Position":
        """
        Adds two positions together.

        Args:
            other (Position): The other position to add.

        Returns:
            Position: The sum of the two positions.
        """
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Position") -> Tuple[int, int]:
        """
        Subtracts one position from another.

        Args:
            other (Position): The position to subtract.

        Returns:
            Tuple[int, int]: The difference between the two positions as a tuple (dx, dy).
        """
        return self.x - other.x, self.y - other.y

    def __hash__(self) -> int:
        """
        Returns the hash value of the position.

        Returns:
            int: The hash value of the position.
        """
        return hash((self.x, self.y))

    def __eq__(self, other: object) -> bool:
        """
        Checks if two positions are equal.

        Args:
            other (object): The other position to compare.

        Returns:
            bool: True if the positions are equal, False otherwise.
        """
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        return False

    def __lt__(self, other: "Position") -> bool:
        """
        Compares two positions lexicographically.

        Args:
            other (Position): The other position to compare.

        Returns:
            bool: True if self is less than other, False otherwise.
        """
        return (self.x, self.y) < (other.x, other.y)

    def distance(self, other: "Position") -> float:
        """
        Calculates the Euclidean distance between two positions.

        Args:
            other (Position): The other position.

        Returns:
            float: The Euclidean distance between the positions.
        """
        dx, dy = self - other
        return (dx**2 + dy**2) ** 0.5


@dataclass(frozen=True)
class Direction(Enum):
    """
    Enumeration representing the possible directions the snake can move. Relative to the forward facing direction of the snake.
    """

    # +x is to the right, +y is downwards. Relative x = 0, y = -1 is forward. Which is default Continuous
    # -x is to the left, -y is upwards. Relative x = 0, y = 1 is backward. Which is forbidden.
    UP = Position(0, -1)  # Default direction moving up
    UP_RIGHT = Position(1, -1)  # Moving diagonal right equal to keypress up and right.
    UP_LEFT = Position(-1, -1)  # Moving diagonal left equal to keypress up and left.
    RIGHT = Position(1, 0)  # Moving right equal to keypress right.
    LEFT = Position(-1, 0)  # Moving left equal to keypress left.
    DOWN = Position(0, 1)  # Moving down equal to keypress down.
    DOWN_RIGHT = Position(
        1, 1
    )  # Moving diagonal right equal to keypress down and right.
    DOWN_LEFT = Position(-1, 1)  # Moving diagonal left equal to keypress down and left.
    # Continue = pick a random direction from the above 8 directions.
    CONTINUE = random.choice(
        [UP, UP_RIGHT, UP_LEFT, RIGHT, LEFT, DOWN, DOWN_RIGHT, DOWN_LEFT]
    )


class Grid:
    """
    Represents the game grid, managing positions, tracking paths, and handling collision detection.

    Attributes:
        width (int): The width of the grid in pixels.
        height (int): The height of the grid in pixels.
        block_size (int): The size of each block in the grid in pixels.
        grid (List[List[int]]): The 2D grid representing the game state, where 0 is empty and 1 is occupied.
        snake (Snake): Reference to the snake object.
        fruit (Fruit): Reference to the fruit object.
    """

    def __init__(
        self,
        width: int,
        height: int,
        block_size: int,
        snake: "Snake",
        fruit: "Fruit",
    ) -> None:
        """
        Initializes the Grid object.

        Args:
            width (int): The width of the grid in pixels.
            height (int): The height of the grid in pixels.
            block_size (int): The size of each block in the grid in pixels.
            snake (Snake): Reference to the snake object.
            fruit (Fruit): Reference to the fruit object.
        """
        self.width: int = width
        self.height: int = height
        self.block_size: int = block_size
        self.grid: List[List[int]] = [
            [0] * (height // block_size) for _ in range(width // block_size)
        ]
        self.snake: "Snake" = snake
        self.fruit: "Fruit" = fruit

        logging.info(
            f"Grid initialized with width: {width}, height: {height}, block size: {block_size}"
        )

    def update_grid(self) -> None:
        """
        Updates the grid based on the current positions of the snake and fruit.
        """
        self.grid = [
            [0] * (self.height // self.block_size)
            for _ in range(self.width // self.block_size)
        ]

        for segment in self.snake.body:
            x, y = segment.x // self.block_size, segment.y // self.block_size
            self.grid[x][y] = 1

        fruit_x, fruit_y = (
            self.fruit.position.x // self.block_size,
            self.fruit.position.y // self.block_size,
        )
        self.grid[fruit_x][fruit_y] = 0

        logging.debug(f"Grid updated with snake positions: {self.snake.body}")

    def is_collision(self, position: Position) -> bool:
        """
        Checks if the given position collides with the snake's body or the grid boundaries.

        Args:
            position (Position): The position to check for collision.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        x, y = position.x // self.block_size, position.y // self.block_size

        if (
            x < 0
            or x >= self.width // self.block_size
            or y < 0
            or y >= self.height // self.block_size
        ):
            logging.debug(f"Collision detected with grid boundaries at {position}")
            return True

        if self.grid[x][y] == 1:
            logging.debug(f"Collision detected with snake body at {position}")
            return True

        return False

    def find_path(
        self, start: Position, end: Position, algorithm: Callable
    ) -> List[Position]:
        """
        Finds a path from the start position to the end position using the specified pathfinding algorithm.

        Args:
            start (Position): The starting position.
            end (Position): The ending position.
            algorithm (Callable): The pathfinding algorithm to use.

        Returns:
            List[Position]: The path from start to end as a list of positions.
        """
        self.update_grid()
        path: List[Position] = algorithm(start, end, self.grid)
        logging.debug(f"Path found from {start} to {end}: {path}")
        return path

    def __str__(self) -> str:
        """
        Returns a string representation of the Grid object.

        Returns:
            str: The string representation of the Grid object.
        """
        return f"Grid(width={self.width}, height={self.height}, block_size={self.block_size})"


class DefaultCollisionDetectionStrategy:
    """
    Default collision detection strategy that checks if a position collides with the snake's body or the grid boundaries.
    """

    def __init__(self, grid: Grid) -> None:
        """
        Initializes the DefaultCollisionDetectionStrategy object.

        Args:
            grid (Grid): Reference to the grid object.
        """
        self.grid: Grid = grid
        logging.info("DefaultCollisionDetectionStrategy initialized")

    def detect_collision(self, position: Position) -> bool:
        """
        Checks if the given position collides with the snake's body or the grid boundaries.

        Args:
            position (Position): The position to check for collision.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        x: int = position.x // self.grid.block_size
        y: int = position.y // self.grid.block_size

        if (
            x < 0
            or x >= self.grid.width // self.grid.block_size
            or y < 0
            or y >= self.grid.height // self.grid.block_size
        ):
            logging.debug(f"Collision detected with grid boundaries at {position}")
            return True

        if self.grid.grid[x][y] == 1:
            logging.debug(f"Collision detected with snake body at {position}")
            return True

        return False

    def __str__(self) -> str:
        """
        Returns a string representation of the DefaultCollisionDetectionStrategy object.

        Returns:
            str: The string representation of the DefaultCollisionDetectionStrategy object.
        """
        return f"DefaultCollisionDetectionStrategy(grid={self.grid})"


class GameLogic:
    """
    Manages the game logic, including game state, score, and game over conditions.

    Attributes:
        grid (Grid): Reference to the grid object.
        snake (Snake): Reference to the snake object.
        fruit (Fruit): Reference to the fruit object.
        score (int): The current score of the game.
        game_over (bool): Flag indicating if the game is over.
    """

    def __init__(self, grid: Grid, snake: "Snake", fruit: "Fruit") -> None:
        """
        Initializes the GameLogic object.

        Args:
            grid (Grid): Reference to the grid object.
            snake (Snake): Reference to the snake object.
            fruit (Fruit): Reference to the fruit object.
        """
        self.grid: Grid = grid
        self.snake: "Snake" = snake
        self.fruit: "Fruit" = fruit
        self.score: int = 0
        self.game_over: bool = False

        logging.info("GameLogic initialized")

    def update(self) -> None:
        """
        Updates the game state by moving the snake, checking for collisions, and handling fruit consumption.
        """
        self.snake.move()

        if self.grid.is_collision(self.snake.body[0]):
            self.game_over = True
            logging.info("Game over due to collision")
            return

        if self.snake.body[0] == self.fruit.position:
            self.snake.grow()
            self.score += self.fruit.points
            self.fruit.relocate()
            logging.info(f"Fruit consumed, score: {self.score}")

    def reset(self) -> None:
        """
        Resets the game state to the initial conditions.
        """
        self.snake.reset()
        self.fruit.relocate()
        self.score = 0
        self.game_over = False
        logging.info("Game reset")

    def __str__(self) -> str:
        """
        Returns a string representation of the GameLogic object.

        Returns:
            str: The string representation of the GameLogic object.
        """
        return f"GameLogic(score={self.score}, game_over={self.game_over})"


class Snake:
    """
    Represents the snake in the game, which moves according to AI decisions, grows by consuming fruits, and avoids collisions.

    Attributes:
        body (deque[Position]): A deque storing the positions of the snake's segments.
        growing (int): A counter to manage the growth of the snake after eating a fruit.
        direction (Position): The current movement direction of the snake.
        fruit (Fruit): A reference to the fruit object in the game.
        ai_controller (AIController): The AI controller that decides the snake's movements.
        grid (Grid): Reference to the grid object for position and collision management.
        collision_detection_strategy (CollisionDetectionStrategy): The strategy used for collision detection.
    """

    def __init__(
        self,
        start_position: Position = Position(
            SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100
        ),
        initial_length: int = 3,
        grid: Optional[Grid] = None,
        collision_detection_strategy=DefaultCollisionDetectionStrategy(Grid),
    ) -> None:
        """
        Initializes the Snake object with a starting position, initial length, reference to the grid, and collision detection strategy.

        Args:
            start_position (Position): The starting position of the snake's head. Defaults to the center of the screen.
            initial_length (int): The initial number of segments of the snake. Defaults to 3.
            grid (Optional[Grid]): Reference to the grid object for position and collision management.
            collision_detection_strategy (CollisionDetectionStrategy): The strategy used for collision detection. Defaults to DefaultCollisionDetectionStrategy.
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
        self.fruit: Optional["Fruit"] = None
        self.ai_controller: Optional[AIController] = None
        self.grid: Optional[Grid] = grid
        self.collision_detection_strategy: CollisionDetectionStrategy = (
            collision_detection_strategy
        )

        # Initialize logging
        logging.basicConfig(
            filename="snake.log",
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.info("Snake initialized")

    def set_fruit(self, fruit: "Fruit") -> None:
        """
        Sets the reference to the fruit object.

        Args:
            fruit (Fruit): The fruit object to link with the snake.
        """
        self.fruit = fruit
        logging.info("Fruit set for the snake")

    def set_ai_controller(self, ai_controller: "AIController") -> None:
        """
        Sets the reference to the AI controller.

        Args:
            ai_controller (AIController): The AI controller to link with the snake.
        """
        self.ai_controller = ai_controller
        logging.info("AI controller set for the snake")

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

    def move(self) -> None:
        """
        Moves the snake based on the AI's decision and checks for collisions.
        """
        if self.ai_controller is None or self.fruit is None or self.grid is None:
            logging.error(
                "AI controller, fruit, or grid not set before moving the snake"
            )
            return

        self.direction = self.ai_controller.decide_action(
            self.body[0],
            self.fruit.position,
        )
        new_head: Position = Position(
            self.body[0].x + self.direction.x * BLOCK_SIZE,
            self.body[0].y + self.direction.y * BLOCK_SIZE,
        )
        if self.grid.is_collision(new_head):
            raise ValueError("Collision detected")
        self.body.appendleft(new_head)
        if not self.growing:
            self.body.pop()
        else:
            self.growing -= 1  # Decrement the growing counter after adding a segment

        # Log movement
        logging.debug(f"Snake moved to {new_head}")

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
            if Position(x, y) in self.body or not self.grid.is_valid_position(
                Position(x, y)
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

    def reset(self) -> None:
        """
        Resets the snake to its initial state.
        """
        self.body.clear()
        self.body.extend(
            [Position(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100)]
            + [
                Position(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100 - i * BLOCK_SIZE)
                for i in range(1, 3)
            ]
        )
        self.growing = 0
        self.direction = Direction.DOWN.value
        logging.info("Snake reset")

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


class Fruit:
    """
    Represents the fruit object in the snake game, which the snake aims to consume.
    The fruit changes its position randomly on the game grid and has a pulsating color effect with a highly detailed graphical star representation.
    """

    def __init__(self, grid: Grid, snake: Optional["Snake"] = None):
        """
        Initializes the Fruit object with a cycling color scheme and a default position.

        Args:
            grid (Grid): The grid object for managing positions and collisions.
            snake (Optional['Snake']): The snake object that will interact with this fruit. Defaults to None.
        """
        self.grid: Grid = grid
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
            new_position: Position = self.grid.get_random_empty_position()
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
        if new_position == Position(
            snake_head.x + snake_direction.x, snake_head.y + snake_direction.y
        ) or new_position == Position(
            snake_head.x - snake_direction.x, snake_head.y - snake_direction.y
        ):
            return True

        # Check if the new position is surrounded by the snake's body
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if self.grid.is_collision(
                Position(
                    new_position.x + dx * BLOCK_SIZE,
                    new_position.y + dy * BLOCK_SIZE,
                )
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

    def __repr__(self) -> str:
        """
        Returns a string representation of the Fruit object.

        Returns:
            str: The string representation of the Fruit object.
        """
        return f"Fruit(position={self.position}, points={self.points})"

    def __str__(self) -> str:
        """
        Returns a readable string representation of the Fruit object.

        Returns:
            str: The readable string representation of the Fruit object.
        """
        return f"Fruit at {self.position} with {self.points} points"


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
                (BLOCK_SIZE, BLOCK_SIZE),  # Diagonal movement
                (BLOCK_SIZE, -BLOCK_SIZE),
                (-BLOCK_SIZE, BLOCK_SIZE),
                (-BLOCK_SIZE, -BLOCK_SIZE),
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

                        # Check for line of sight to the parent's parent
                        if current in came_from and ThetaStar.line_of_sight(
                            grid, came_from[current], neighbor
                        ):
                            came_from[neighbor] = came_from[current]
                            g_score[neighbor] = g_score[came_from[current]] + came_from[
                                current
                            ].distance(neighbor)
                            f_score[neighbor] = g_score[neighbor] + neighbor.distance(
                                goal
                            )

        return []

    @staticmethod
    def hierarchical_theta_star(
        start: Position,
        goal: Position,
        grid: List[List[int]],
        levels: int = 3,  # Increased levels for more granular pathfinding
    ) -> List[Position]:
        """
        Perform hierarchical pathfinding using the Theta* algorithm to efficiently manage larger grids.

        Args:
            start (Position): The start coordinate (x, y).
            goal (Position): The goal coordinate (x, y).
            grid (List[List[int]]): The game grid where 1 represents obstacles (snake body).
            levels (int): The number of hierarchical levels to use. Defaults to 3.

        Returns:
            List[Position]: The path from start to goal as a list of coordinates.
        """
        if levels <= 1:
            return ThetaStar.theta_star(start, goal, grid)

        # Create a coarser grid for higher-level pathfinding
        coarse_factor: int = 2 ** (levels - 1)
        coarse_grid: List[List[int]] = [
            [
                min(
                    grid[i * coarse_factor + dx][j * coarse_factor + dy]
                    for dx in range(coarse_factor)
                    for dy in range(coarse_factor)
                )
                for j in range(len(grid[0]) // coarse_factor)
            ]
            for i in range(len(grid) // coarse_factor)
        ]

        # Find a high-level path using the coarser grid
        coarse_path: List[Position] = ThetaStar.hierarchical_theta_star(
            Position(start.x // coarse_factor, start.y // coarse_factor),
            Position(goal.x // coarse_factor, goal.y // coarse_factor),
            coarse_grid,
            levels - 1,
        )

        # Refine the path using the original grid
        refined_path: List[Position] = []
        for i in range(len(coarse_path) - 1):
            refined_path.extend(
                ThetaStar.theta_star(
                    Position(
                        coarse_path[i].x * coarse_factor,
                        coarse_path[i].y * coarse_factor,
                    ),
                    Position(
                        coarse_path[i + 1].x * coarse_factor,
                        coarse_path[i + 1].y * coarse_factor,
                    ),
                    grid,
                )
            )
        refined_path.append(goal)

        return refined_path

    @staticmethod
    def adaptive_theta_star(
        start: Position,
        goal: Position,
        grid: List[List[int]],
        snake: Snake,
        fruit: Fruit,
        max_depth: int = 10,  # Maximum depth for recursive calls
    ) -> List[Position]:
        """
        Perform adaptive pathfinding using the Theta* algorithm, considering potential snake growth and fruit spawns.

        Args:
            start (Position): The start coordinate (x, y).
            goal (Position): The goal coordinate (x, y).
            grid (List[List[int]]): The game grid where 1 represents obstacles (snake body).
            snake (Snake): The snake object, used for predicting future body positions.
            fruit (Fruit): The fruit object, used for predicting potential spawn locations.
            max_depth (int): The maximum depth for recursive calls. Defaults to 10.

        Returns:
            List[Position]: The path from start to goal as a list of coordinates.
        """
        if max_depth <= 0:
            return ThetaStar.theta_star(start, goal, grid)

        # Predict potential snake body positions after eating the fruit
        potential_snake_positions: Set[Position] = set(snake.body)
        potential_snake_positions.update(
            snake.get_potential_positions(fruit.position, 5)  # Predict 5 steps ahead
        )

        # Predict potential fruit spawn locations
        potential_fruit_positions: Set[Position] = set(
            fruit.get_potential_positions(grid, 3)
        )  # Predict 3 potential spawns

        # Create a modified grid considering potential snake and fruit positions
        modified_grid: List[List[int]] = [
            [
                (
                    1
                    if Position(x, y) in potential_snake_positions
                    or Position(x, y) in potential_fruit_positions
                    else grid[x][y]
                )
                for y in range(len(grid[0]))
            ]
            for x in range(len(grid))
        ]

        # Find an adaptive path using the modified grid
        adaptive_path: List[Position] = ThetaStar.adaptive_theta_star(
            start, goal, modified_grid, snake, fruit, max_depth - 1
        )

        return adaptive_path

    def __str__(self) -> str:
        """
        Provide a human-readable representation of the ThetaStar pathfinding algorithm's capabilities.
        """
        return "ThetaStar pathfinding algorithm with dynamic snake body adjustment, hierarchical pathfinding, and adaptive pathfinding based on potential snake growth and fruit spawns."


class AIController:
    """
    Manages AI decision-making using Q-learning, enhanced pathfinding algorithms, and neural network integration.
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
        neural_network (keras.Model): A neural network to enhance decision-making.
        pathfinding_algorithms (List[Callable]): List of pathfinding algorithms, ranked by priority.
        learning_algorithms (List[Callable]): List of learning algorithms, ranked by priority.
        game_grid (GameGrid): Reference to the game grid for centralized position tracking and collision detection.
    """

    def __init__(self, snake: Snake, fruit: Fruit, game_grid: Grid):
        """
        Initialize the AIController with references to the snake, fruit, and game grid, and setup the Q-table and other necessary structures.

        Args:
            snake (Snake): Reference to the snake object.
            fruit (Fruit): Reference to the fruit object.
            game_grid (GameGrid): Reference to the game grid object.
        """
        self.snake: Snake = snake
        self.fruit: Fruit = fruit
        self.game_grid: Grid = game_grid
        self.q_table: np.ndarray = np.zeros(
            (
                SCREEN_WIDTH // BLOCK_SIZE,
                SCREEN_HEIGHT // BLOCK_SIZE,
                8,
            ),  # Increased action space
            dtype=np.float32,
        )
        self.action_map: List[Position] = [
            Position(0, -1),
            Position(0, 1),
            Position(-1, 0),
            Position(1, 0),
            Position(1, 1),  # Diagonal movements
            Position(1, -1),
            Position(-1, 1),
            Position(-1, -1),
        ]
        self.state_cache: Dict[Tuple[Position, Position], List[Position]] = {}
        self.lock: threading.Lock = threading.Lock()
        self.neural_network: keras.Model = self._initialize_neural_network()
        self.pathfinding_algorithms: List[Callable] = [
            self.advanced_adaptive_theta_star,  # Highest priority
            self.advanced_hierarchical_theta_star,
            self.advanced_theta_star,
            self.decide_by_neural_network,
        ]
        self.learning_algorithms: List[Callable] = [
            self.decide_by_q_learning,
            self.decide_by_neural_network,
        ]

    def _initialize_neural_network(self) -> keras.Model:
        """
        Initialize a complex neural network for decision-making.

        Returns:
            keras.Model: A complex neural network model.
        """
        model = keras.Sequential(
            [
                keras.Input(shape=(8,)),  # Increased input dimensions
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(512, activation="relu"),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(
                    8, activation="linear"
                ),  # Increased output dimensions
            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
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
        path: List[Position] = []  # Default to an empty path
        with self.lock:
            if (state, goal) in self.state_cache:
                path = self.state_cache[(state, goal)]
            else:
                for pathfinding_algorithm in self.pathfinding_algorithms:
                    path = pathfinding_algorithm(
                        state, goal, self.game_grid, self.snake, self.fruit
                    )
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
            return self.action_map[randint(0, 7)]  # Increased action space
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
        x_train = np.array(
            [
                state.x,
                state.y,
                goal.x,
                goal.y,
                self.snake.length,
                self.game_grid.width,
                self.game_grid.height,
                len(path),
            ]
        ).reshape(1, -1)
        y_train = np.zeros((1, 8))

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
        x_input: np.ndarray = np.array(
            [
                state.x,
                state.y,
                goal.x,
                goal.y,
                self.snake.length,
                self.game_grid.width,
                self.game_grid.height,
                len(self.snake.body),
            ],
            dtype=np.float32,
        ).reshape(1, -1)

        with self.lock:
            q_values: np.ndarray = self.neural_network.predict(x_input)[0]

        action_index: int = int(np.argmax(q_values))
        action: Optional[Position] = (
            self.action_map[action_index]
            if 0 <= action_index < len(self.action_map)
            else None
        )

        if action is not None:
            logging.info(
                f"Neural network decided on action {action} for state {state} and goal {goal}"
            )
        else:
            logging.warning(
                f"Neural network failed to decide on an action for state {state} and goal {goal}"
            )

        return action

    def update_algorithm_priorities(self, game_state: Dict[str, Any]) -> None:
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

        # TODO: Implement more sophisticated priority updates based on game state analysis

    def advanced_theta_star(
        self,
        start: Position,
        goal: Position,
        game_grid: GameGrid,
        snake: Snake,
        fruit: Fruit,
        max_depth: int = 20,
    ) -> List[Position]:
        """
        An advanced version of the Theta* pathfinding algorithm that considers potential snake body positions and fruit spawns.

        Args:
            start (Position): The start coordinate (x, y).
            goal (Position): The goal coordinate (x, y).
            game_grid (GameGrid): The game grid object for centralized position tracking and collision detection.
            snake (Snake): The snake object, used for predicting future body positions.
            fruit (Fruit): The fruit object, used for predicting potential spawn locations.
            max_depth (int): The maximum depth for recursive calls. Defaults to 20.

        Returns:
            List[Position]: The path from start to goal as a list of coordinates.
        """
        if max_depth <= 0:
            return ThetaStar.theta_star(start, goal, game_grid.grid)

        # Predict potential snake body positions after eating the fruit
        potential_snake_positions: Set[Position] = set(snake.body)
        potential_snake_positions.update(
            snake.get_potential_positions(fruit.position, 10)  # Predict 10 steps ahead
        )

        # Predict potential fruit spawn locations
        potential_fruit_positions: Set[Position] = set(
            fruit.get_potential_positions(game_grid.grid, 5)
        )  # Predict 5 potential spawns

        # Create a modified grid considering potential snake and fruit positions
        modified_grid: List[List[int]] = [
            [
                int(
                    Position(x, y) in potential_snake_positions
                    or Position(x, y) in potential_fruit_positions
                    or game_grid.grid[x][y] == 1
                )
                for y in range(len(game_grid.grid[0]))
            ]
            for x in range(len(game_grid.grid))
        ]

        # Find an advanced path using the modified grid
        advanced_path: List[Position] = self.advanced_adaptive_theta_star(
            start, goal, modified_grid, snake, fruit, max_depth - 1
        )

        return advanced_path

    def advanced_adaptive_theta_star(
        self,
        start: Position,
        goal: Position,
        grid: List[List[int]],
        snake: Snake,
        fruit: Fruit,
        max_depth: int = 20,
    ) -> List[Position]:
        """
        An advanced version of the Adaptive Theta* pathfinding algorithm that dynamically adjusts the snake's body and considers potential fruit spawns.

        Args:
            start (Position): The start coordinate (x, y).
            goal (Position): The goal coordinate (x, y).
            grid (List[List[int]]): The game grid where 1 represents obstacles (snake body).
            snake (Snake): The snake object, used for predicting future body positions.
            fruit (Fruit): The fruit object, used for predicting potential spawn locations.
            max_depth (int): The maximum depth for recursive calls. Defaults to 20.

        Returns:
            List[Position]: The path from start to goal as a list of coordinates.
        """
        if max_depth <= 0:
            return ThetaStar.theta_star(start, goal, grid)

        # Predict potential snake body positions after eating the fruit
        potential_snake_positions: Set[Position] = set(snake.body)
        potential_snake_positions.update(
            snake.get_potential_positions(fruit.position, 10)  # Predict 10 steps ahead
        )

        # Predict potential fruit spawn locations
        potential_fruit_positions: Set[Position] = set(
            fruit.get_potential_positions(grid, 5)
        )  # Predict 5 potential spawns

        # Create a modified grid considering potential snake and fruit positions
        modified_grid: List[List[int]] = [
            [
                int(
                    Position(x, y) in potential_snake_positions
                    or Position(x, y) in potential_fruit_positions
                    or grid[x][y] == 1
                )
                for y in range(len(grid[0]))
            ]
            for x in range(len(grid))
        ]

        # Find an advanced adaptive path using the modified grid
        advanced_adaptive_path: List[Position] = ThetaStar.adaptive_theta_star(
            start, goal, modified_grid, snake, fruit, max_depth - 1
        )

        return advanced_adaptive_path

    def advanced_hierarchical_theta_star(
        self,
        start: Position,
        goal: Position,
        game_grid: GameGrid,
        snake: Snake,
        fruit: Fruit,
        max_depth: int = 20,
    ) -> List[Position]:
        """
        An advanced version of the Hierarchical Theta* pathfinding algorithm that considers potential snake body positions and fruit spawns.

        Args:
            start (Position): The start coordinate (x, y).
            goal (Position): The goal coordinate (x, y).
            game_grid (GameGrid): The game grid object for centralized position tracking and collision detection.
            snake (Snake): The snake object, used for predicting future body positions.
            fruit (Fruit): The fruit object, used for predicting potential spawn locations.
            max_depth (int): The maximum depth for recursive calls. Defaults to 20.

        Returns:
            List[Position]: The path from start to goal as a list of coordinates.
        """
        if max_depth <= 0:
            return ThetaStar.theta_star(start, goal, game_grid.grid)

        # Predict potential snake body positions after eating the fruit
        potential_snake_positions: Set[Position] = set(snake.body)
        potential_snake_positions.update(
            snake.get_potential_positions(fruit.position, 10)  # Predict 10 steps ahead
        )

        # Predict potential fruit spawn locations
        potential_fruit_positions: Set[Position] = set(
            fruit.get_potential_positions(game_grid.grid, 5)
        )  # Predict 5 potential spawns

        # Create a hierarchical grid representation
        hierarchical_grid: List[List[List[int]]] = [
            [
                [
                    int(
                        Position(x, y) in potential_snake_positions
                        or Position(x, y) in potential_fruit_positions
                        or game_grid.grid[x][y] == 1
                    )
                    for y in range(len(game_grid.grid[0]))
                ]
                for x in range(len(game_grid.grid))
            ]
            for _ in range(3)  # 3 levels of hierarchy
        ]

        # Find an advanced hierarchical path using the hierarchical grid
        advanced_hierarchical_path: List[Position] = (
            self.advanced_hierarchical_theta_star_recursive(
                start, goal, hierarchical_grid, snake, fruit, max_depth - 1
            )
        )

        return advanced_hierarchical_path

    def advanced_hierarchical_theta_star_recursive(
        self,
        start: Position,
        goal: Position,
        hierarchical_grid: List[List[List[int]]],
        snake: Snake,
        fruit: Fruit,
        max_depth: int = 20,
    ) -> List[Position]:
        """
        Recursive function for the advanced hierarchical Theta* pathfinding algorithm.

        Args:
            start (Position): The start coordinate (x, y).
            goal (Position): The goal coordinate (x, y).
            hierarchical_grid (List[List[List[int]]]): The hierarchical grid representation.
            snake (Snake): The snake object, used for predicting future body positions.
            fruit (Fruit): The fruit object, used for predicting potential spawn locations.
            max_depth (int): The maximum depth for recursive calls. Defaults to 20.

        Returns:
            List[Position]: The path from start to goal as a list of coordinates.
        """
        if max_depth <= 0 or len(hierarchical_grid) == 0:
            return ThetaStar.theta_star(start, goal, hierarchical_grid[0])

        # Find a path using the current level of the hierarchical grid
        path: List[Position] = ThetaStar.theta_star(start, goal, hierarchical_grid[0])

        if len(path) <= 2:
            return path

        # Recursively refine the path using lower levels of the hierarchy
        refined_path: List[Position] = []
        for i in range(len(path) - 1):
            sub_path: List[Position] = self.advanced_hierarchical_theta_star_recursive(
                path[i], path[i + 1], hierarchical_grid[1:], snake, fruit, max_depth - 1
            )
            refined_path.extend(sub_path[:-1])
        refined_path.append(path[-1])

        return refined_path

    def run(self) -> None:
        """
        Run the AI snake game.
        """
        while not self.game_over:
            # Choose the next action based on the current game state
            state: Position = self.snake.head
            goal: Position = self.fruit.position

            action: Optional[Position] = None
            path: List[Position] = []

            # Use pathfinding algorithms first
            for pathfinding_func in self.pathfinding_algorithms:
                try:
                    path = pathfinding_func(
                        state, goal, self.game_grid, self.snake, self.fruit
                    )
                    if path:
                        action = path[1] - state
                        break
                except Exception as e:
                    logging.error(
                        f"Error in pathfinding function {pathfinding_func.__name__}: {e}"
                    )

            # If pathfinding fails, use learning algorithms
            if action is None:
                for learning_func in self.learning_algorithms:
                    try:
                        action = learning_func(state, goal)
                        if action is not None:
                            break
                    except Exception as e:
                        logging.error(
                            f"Error in learning function {learning_func.__name__}: {e}"
                        )

            # If all else fails, use the neural network
            if action is None:
                action = self.decide_by_neural_network(state, goal)

            if action is None:
                # No valid action found, game over
                self.game_over = True
                break

            # Update game state based on the chosen action
            next_state: Position = state + action
            self.snake.move(action)
            self.game_grid.update(self.snake, self.fruit)

            # Check for collision with walls or snake body
            if (
                not (0 <= next_state.x < self.game_grid.width)
                or not (0 <= next_state.y < self.game_grid.height)
                or self.game_grid.grid[next_state.x][next_state.y] == 1
            ):
                self.game_over = True
                reward: float = -1.0
            elif next_state == self.fruit.position:
                # Snake ate the fruit
                self.snake.length += 1
                self.snake.score += 1
                reward = 1.0
                self.fruit.respawn(self.game_grid.grid)
            else:
                reward = 0.0

            # Train the learning models
            if path:
                self.train_q_learning(state, action, reward, next_state)
                self.train_neural_network(state, goal, path)

            # Update algorithm priorities based on game state
            game_state: Dict[str, Any] = {
                "step": self.snake.score,
                "snake_length": self.snake.length,
                "fruit_eaten": self.snake.score,
            }
            self.update_algorithm_priorities(game_state)

            # Delay to control game speed
            time.sleep(0.1)

        logging.info(f"Game over. Final score: {self.snake.score}")
