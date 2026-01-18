import pygame as pg
import sys
from random import randint, seed
from collections import deque, defaultdict
from typing import List, Tuple, Deque, Optional, Set, Dict
import logging
import math

# Setup comprehensive logging
logging.basicConfig(
    filename="snake_game.log",
    level=logging.DEBUG,  # Increased logging level for more detailed output
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Constants for game dimensions and block size
SCREEN_WIDTH: int = 600
SCREEN_HEIGHT: int = 400
BLOCK_SIZE: int = 20

# Seed for consistent random behavior
seed(0)

# Initialize pygame
pg.init()
window = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pg.display.set_caption("Dynamic Hamiltonian Snake Game")


class Grid:
    """
    Manages the grid system of the game, tracking positions of the snake and fruit.
    """

    def __init__(self, width: int, height: int, block_size: int) -> None:
        self.width = width
        self.height = height
        self.block_size = block_size
        self.snake_positions: Set[Tuple[int, int]] = set()
        self.fruit_position: Tuple[int, int] = (0, 0)

    def update_snake_position(self, new_positions: Deque[Tuple[int, int]]) -> None:
        self.snake_positions = set(new_positions)

    def update_fruit_position(self, new_position: Tuple[int, int]) -> None:
        self.fruit_position = new_position

    def is_position_occupied(self, position: Tuple[int, int]) -> bool:
        return position in self.snake_positions or position == self.fruit_position


class Fruit:
    """
    Represents a fruit in the snake game, randomly placed on the screen avoiding the snake.
    """

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.color: pg.Color = pg.Color(139, 0, 0)  # Deep red for visibility
        self.radius: int = 10
        self.position: Tuple[int, int] = (0, 0)
        self.relocate()

    def draw(self) -> None:
        """Draws the fruit on the game screen."""
        pg.draw.circle(
            window,
            self.color,
            (self.position[0] + self.radius, self.position[1] + self.radius),
            self.radius,
        )

    def relocate(self) -> None:
        """
        Relocates the fruit to a random position not occupied by the snake.
        """
        while True:
            new_x: int = (
                randint(0, (self.grid.width // self.grid.block_size) - 1)
                * self.grid.block_size
            )
            new_y: int = (
                randint(0, (self.grid.height // self.grid.block_size) - 1)
                * self.grid.block_size
            )
            new_position: Tuple[int, int] = (new_x, new_y)
            if not self.grid.is_position_occupied(new_position):
                self.position = new_position
                self.grid.update_fruit_position(new_position)
                break
        logging.info(f"Fruit relocated to {self.position}")


class Snake:
    """
    Manages the snake's state and behavior, including movement, growth, and collision detection.
    """

    def __init__(self, grid: Grid, fruit: Fruit) -> None:
        self.grid = grid
        self.fruit = fruit
        self.body: Deque[Tuple[int, int]] = deque([(20, 20), (40, 20), (60, 20)])
        self.growing: int = 0
        self.score: int = 0
        self.grid.update_snake_position(self.body)

    def draw(self) -> None:
        """Draws each segment of the snake."""
        for segment in self.body:
            pg.draw.rect(
                window,
                pg.Color(220, 20, 60),
                pg.Rect(
                    segment[0], segment[1], self.grid.block_size, self.grid.block_size
                ),
            )

    def move(self) -> None:
        """
        Moves the snake according to a calculated path and handles potential collisions.
        """
        next_position: Tuple[int, int] = self.calculate_next_position()

        if next_position in self.body:
            logging.error("Collision detected; restarting game.")
            self.restart_game()
        else:
            self.body.appendleft(next_position)
            if not self.growing:
                self.body.pop()
            else:
                self.growing -= 1
            self.grid.update_snake_position(self.body)
            logging.info(f"Snake moved to {next_position}")

    def calculate_next_position(self) -> Tuple[int, int]:
        """
        Calculates the next position of the snake using the Theta* pathfinding algorithm.
        """
        path = theta_star_path(self.body[0], self.fruit.position, self.body, self.grid)
        return (
            path[0] if path else self.body[0]
        )  # Continue in the same direction if no path found

    def grow(self) -> None:
        """Increases the size of the snake and updates the score."""
        self.growing += 3
        self.score += 10
        logging.info("Snake grows. Score updated.")

    def restart_game(self) -> None:
        """Restarts the game following a collision or similar event."""
        self.body = deque([(20, 20), (40, 20), (60, 20)])
        self.score = 0
        self.growing = 0
        self.grid.update_snake_position(self.body)
        logging.info("Game restarted.")


class Pathfinding:
    """
    Implements advanced Theta* pathfinding algorithm for the snake to find the most efficient path.
    """

    def __init__(self, grid: Grid) -> None:
        self.grid = grid

    def theta_star_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        body: Deque[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        Theta* pathfinding algorithm to find the most efficient path from start to goal avoiding the snake's body.
        :param start: Starting node of the path.
        :param goal: Goal node of the path.
        :param body: Current snake body to avoid.
        :return: List of tuples representing the path from start to goal.
        """
        open_set: Set[Tuple[int, int]] = {start}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        f_score: Dict[Tuple[int, int], int] = {
            start: self.euclidean_heuristic(start, goal)
        }

        while open_set:
            current: Tuple[int, int] = min(open_set, key=lambda o: f_score[o])
            if current == goal:
                return self.reconstruct_path(came_from, current)

            open_set.remove(current)
            for neighbor in self.neighbors(current, body):
                tentative_g_score: int = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.euclidean_heuristic(
                        neighbor, goal
                    )
                    open_set.add(neighbor)

        return []

    def reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Reconstructs the path from start to goal using the `came_from` mapping.
        :param came_from: Map of each node to its predecessor.
        :param current: Current node to trace back from.
        :return: List of tuples representing the reconstructed path.
        """
        path: List[Tuple[int, int]] = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)  # Add the start node
        path.reverse()
        return path

    def neighbors(
        self, node: Tuple[int, int], body: Deque[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Identifies valid adjacent nodes not part of the snake's body.
        :param node: Current node for finding neighbors.
        :param body: Current snake body to avoid.
        :return: List of tuples representing valid adjacent nodes.
        """
        directions: List[Tuple[int, int]] = [
            (0, self.grid.block_size),
            (0, -self.grid.block_size),
            (self.grid.block_size, 0),
            (-self.grid.block_size, 0),
        ]
        result: List[Tuple[int, int]] = []
        for dx, dy in directions:
            x, y = node[0] + dx, node[1] + dy
            if (
                0 <= x < self.grid.width
                and 0 <= y < self.grid.height
                and (x, y) not in body
            ):
                result.append((x, y))
        return result

    def euclidean_heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculates the Euclidean distance heuristic for Theta* between two nodes.
        :param a: Current node.
        :param b: Goal node.
        :return: Euclidean distance as an integer.
        """
        return int(
            math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / self.grid.block_size
        )


class AIController:
    """
    AI Controller that acts as the "player" controlling the snake using pathfinding.
    """

    def __init__(self, snake: Snake, pathfinding: Pathfinding) -> None:
        self.snake = snake
        self.pathfinding = pathfinding

    def make_decision(self) -> None:
        """
        Makes decisions for the snake's next move based on pathfinding.
        """
        self.snake.move()


class Rendering:
    """
    Handles all rendering operations for the game.
    """

    def __init__(self, window, snake: Snake, fruit: Fruit) -> None:
        self.window = window
        self.snake = snake
        self.fruit = fruit

    def render(self) -> None:
        """
        Renders all game objects.
        """
        self.window.fill(pg.Color(0, 0, 0))  # Clear screen with black
        self.snake.draw()
        self.fruit.draw()
        pg.display.flip()  # Update the display


def main():
    """
    Main game loop.
    """
    running = True
    clock = pg.time.Clock()

    grid = Grid(SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE)
    fruit = Fruit(grid)
    snake = Snake(grid, fruit)
    pathfinding = Pathfinding(grid)
    ai_controller = AIController(snake, pathfinding)
    renderer = Rendering(window, snake, fruit)

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False

        ai_controller.make_decision()
        renderer.render()
        clock.tick(10)  # Limit to 10 frames per second

        if snake.score >= 1000:  # Example condition for automatic restart
            snake.restart_game()

    pg.quit()
    sys.exit()


if __name__ == "__main__":
    main()
