import pygame as pg
import sys
from random import randint, seed
from collections import deque, defaultdict
from typing import List, Tuple, Deque, Optional, Set, Dict
import logging

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


class Fruit:
    """
    Represents a fruit in the snake game, randomly placed on the screen avoiding the snake.
    """

    def __init__(self) -> None:
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

    def relocate(self, exclude: Optional[List[Tuple[int, int]]] = None) -> None:
        """
        Relocates the fruit to a random position not occupied by the snake.
        :param exclude: Coordinates to avoid placing the fruit on.
        """
        if exclude is None:
            exclude = []
        while True:
            new_x: int = randint(0, (SCREEN_WIDTH // BLOCK_SIZE) - 1) * BLOCK_SIZE
            new_y: int = randint(0, (SCREEN_HEIGHT // BLOCK_SIZE) - 1) * BLOCK_SIZE
            new_position: Tuple[int, int] = (new_x, new_y)
            if new_position not in exclude:
                self.position = new_position
                break
        logging.info(f"Fruit relocated to {self.position}")


class Snake:
    """
    Manages the snake's state and behavior, including movement, growth, and collision detection.
    """

    def __init__(self, fruit: Fruit) -> None:
        self.body: Deque[Tuple[int, int]] = deque([(20, 20), (40, 20), (60, 20)])
        self.growing: int = 0
        self.score: int = 0
        self.path_cache: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.fruit = fruit

    def draw(self) -> None:
        """Draws each segment of the snake."""
        for segment in self.body:
            pg.draw.rect(
                window,
                pg.Color(220, 20, 60),
                pg.Rect(segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE),
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
            logging.info(f"Snake moved to {next_position}")

    def calculate_next_position(self) -> Tuple[int, int]:
        """
        Calculates the next position of the snake using the A* pathfinding algorithm.
        """
        path = a_star_path(self.body[0], self.fruit.position, self.body)
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
        logging.info("Game restarted.")


def a_star_path(
    start: Tuple[int, int], goal: Tuple[int, int], body: Deque[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Implements the A* algorithm to find the most efficient path from start to goal avoiding the snake's body.
    :param start: Starting node of the path.
    :param goal: Goal node of the path.
    :param body: Current snake body to avoid.
    :return: List of tuples representing the path from start to goal.
    """
    open_set: Set[Tuple[int, int]] = {start}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], int] = {start: 0}
    f_score: Dict[Tuple[int, int], int] = {start: heuristic(start, goal)}

    while open_set:
        current: Tuple[int, int] = min(open_set, key=lambda o: f_score[o])
        if current == goal:
            return reconstruct_path(came_from, current)

        open_set.remove(current)
        for neighbor in neighbors(current, body):
            tentative_g_score: int = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                open_set.add(neighbor)

    return []


def reconstruct_path(
    came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]
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
    node: Tuple[int, int], body: Deque[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Identifies valid adjacent nodes not part of the snake's body.
    :param node: Current node for finding neighbors.
    :param body: Current snake body to avoid.
    :return: List of tuples representing valid adjacent nodes.
    """
    directions: List[Tuple[int, int]] = [
        (0, BLOCK_SIZE),
        (0, -BLOCK_SIZE),
        (BLOCK_SIZE, 0),
        (-BLOCK_SIZE, 0),
    ]
    result: List[Tuple[int, int]] = []
    for dx, dy in directions:
        x, y = node[0] + dx, node[1] + dy
        if 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT and (x, y) not in body:
            result.append((x, y))
    return result


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    Calculates the Manhattan distance heuristic for A* between two nodes.
    :param a: Current node.
    :param b: Goal node.
    :return: Manhattan distance as an integer.
    """
    return abs(a[0] - b[0]) // BLOCK_SIZE + abs(a[1] - b[1]) // BLOCK_SIZE


def main() -> None:
    """
    Main function to run the game loop, handling initialization and user interactions.
    """
    fruit = Fruit()
    snake = Snake(fruit)
    clock = pg.time.Clock()
    running: bool = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                logging.info("Game quit by user.")
        snake.move()
        window.fill(pg.Color(0, 0, 0))
        snake.draw()
        fruit.draw()
        if snake.body[0] == fruit.position:
            snake.grow()
            fruit.relocate(exclude=[seg for seg in snake.body])
        pg.display.flip()
        clock.tick(120)
    pg.quit()
    sys.exit()


if __name__ == "__main__":
    main()
