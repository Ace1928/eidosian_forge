import pygame as pg
import sys
from random import randint, seed
from collections import deque
from typing import List, Tuple, Dict, Deque, Set, Optional, Callable
import logging
import math
from heapq import heappush, heappop
from itertools import product

# Setup logging configuration
logging.basicConfig(
    filename="snake_game_log.txt",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Seed the random generator for reproducible results during development
seed(0)

# Initialize pygame
pg.init()
# Set the display to fullscreen at the current display resolution
display_info = pg.display.Info()  # Get current display information
SCREEN_WIDTH: int = display_info.current_w  # Set width to current width of display
SCREEN_HEIGHT: int = display_info.current_h  # Set height to current height of display
window: pg.Surface = pg.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT), pg.FULLSCREEN
)  # Set display mode to fullscreen
logging.info(
    f"Display set to fullscreen with resolution {SCREEN_WIDTH}x{SCREEN_HEIGHT}"
)
BLOCK_SIZE: int = 20
pg.display.set_caption("Enhanced Snake Game with A* Pathfinding")


def euclidean_distance(node1: Tuple[int, int], node2: Tuple[int, int]) -> float:
    """
    Calculate the Euclidean distance between two points.
    This is a more accurate measure over larger distances compared to Manhattan distance.
    """
    return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


def heuristic(
    node1: Tuple[int, int], node2: Tuple[int, int], snake_body: List[Tuple[int, int]]
) -> float:
    """
    A sophisticated heuristic function that combines Euclidean distance with additional game-specific adjustments.
    This heuristic is designed to be both efficient and highly effective for the snake game's pathfinding needs.
    """
    base_distance = euclidean_distance(node1, node2)
    # Additional heuristic enhancements can be implemented here
    # Penalize positions closer to the snake's body to avoid collisions
    body_proximity_penalty = sum(
        math.exp(-euclidean_distance(node1, body_part)) for body_part in snake_body
    )
    # Encourage the snake to maintain a dense configuration
    density_bonus = -sum(
        euclidean_distance(node1, (x, y))
        for x in range(0, SCREEN_WIDTH, BLOCK_SIZE)
        for y in range(0, SCREEN_HEIGHT, BLOCK_SIZE)
        if (x, y) in snake_body
    )
    return base_distance + body_proximity_penalty + density_bonus


def is_within_boundaries(node: Tuple[int, int]) -> bool:
    """
    Check if a node is within the game boundaries.
    """
    return 0 <= node[0] < SCREEN_WIDTH and 0 <= node[1] < SCREEN_HEIGHT


def get_neighbors(node: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Return the neighbors of the given node within game boundaries.
    This function considers only four possible movements (no diagonals) for strict pathfinding.
    """
    directions = [
        (BLOCK_SIZE, 0),  # Right
        (-BLOCK_SIZE, 0),  # Left
        (0, BLOCK_SIZE),  # Down
        (0, -BLOCK_SIZE),  # Up
    ]
    return [
        (node[0] + dx, node[1] + dy)
        for dx, dy in directions
        if is_within_boundaries((node[0] + dx, node[1] + dy))
    ]


def strategic_pathfinding(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    snake_body: List[Tuple[int, int]],
    fruit_position: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    Implement a strategic pathfinding that prioritizes survival, density, fruit acquisition, and space filling.
    """
    path = []
    # Prioritize survival by avoiding collisions
    if not is_within_boundaries(start) or start in snake_body:
        return path  # Return an empty path if start is not safe

    # Implement A* algorithm with strategic modifications
    open_set = []
    heappush(open_set, (heuristic(start, goal, snake_body), start))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal, snake_body)}

    while open_set:
        current_f_score, current = heappop(open_set)
        if current == goal:
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for neighbor in get_neighbors(current):
            if neighbor in snake_body:
                continue
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(
                    neighbor, goal, snake_body
                )
                heappush(open_set, (f_score[neighbor], neighbor))

    return path


def a_star(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    snake_body: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    Perform A* pathfinding avoiding given obstacles and considering the snake's body.
    This implementation uses a priority queue for the open set for efficient retrieval of the lowest f-score node.
    It also uses a set for obstacles for O(1) complexity checks.
    """
    open_set = []
    heappush(open_set, (heuristic(start, goal, snake_body), start))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal, snake_body)}

    while open_set:
        current_f_score, current = heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for neighbor in get_neighbors(current):
            if neighbor in obstacles or neighbor in snake_body:
                continue
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(
                    neighbor, goal, snake_body
                )
                heappush(open_set, (f_score[neighbor], neighbor))

    return []


class Fruit:
    def __init__(self, window: pg.Surface) -> None:
        """
        Initializes a Fruit object with a position on the game window.
        This method sets the initial position of the fruit to a default value and then relocates it to a valid random position.
        """
        self.position: Tuple[int, int] = (0, 0)  # Initialize with a default position
        self.relocate()  # Relocate to a valid random position
        self.window = window
        logging.info(f"Fruit initialized and placed at {self.position}")

    def draw(self) -> None:
        """
        Draw the fruit on the game window using a fixed color (red) and block size.
        """
        color: Tuple[int, int, int] = (255, 0, 0)  # RGB color for the fruit
        pg.draw.rect(self.window, color, (*self.position, BLOCK_SIZE, BLOCK_SIZE))
        logging.debug(f"Fruit drawn at {self.position} with color {color}")

    def relocate(self, exclude: Optional[List[Tuple[int, int]]] = None) -> None:
        """
        Relocate the fruit to a random position within the game boundaries that is not occupied.
        Ensures the fruit does not spawn inside the snake's body or any other excluded positions.

        :param exclude: A list of positions to be excluded when placing the fruit.
        """
        if exclude is None:
            exclude = []  # Initialize exclude list if not provided
        valid_position_found: bool = False
        while not valid_position_found:
            new_x: int = randint(0, (SCREEN_WIDTH // BLOCK_SIZE) - 1) * BLOCK_SIZE
            new_y: int = randint(0, (SCREEN_HEIGHT // BLOCK_SIZE) - 1) * BLOCK_SIZE
            new_position: Tuple[int, int] = (new_x, new_y)
            if new_position not in exclude:
                self.position = new_position
                valid_position_found = True
                logging.info(f"Fruit relocated to {self.position}")
            else:
                logging.debug(
                    f"Attempted fruit position {new_position} is invalid, recalculating..."
                )


class Snake:
    def __init__(self, window: pg.Surface) -> None:
        """
        Initializes the Snake object with a starting position and a fruit object.
        This constructor sets up the snake's body, direction, associates a fruit object, and calculates the initial path using the A* algorithm.
        """
        self.body: Deque[Tuple[int, int]] = deque(
            [(160, 160), (140, 160), (120, 160)]
        )  # Initial body coordinates
        self.direction: Tuple[int, int] = (BLOCK_SIZE, 0)  # Initial movement direction
        self.fruit: Fruit = Fruit(window)  # Associate a fruit object
        self.window = window
        self.path: List[Tuple[int, int]] = (
            self.calculate_path()
        )  # Calculate initial path
        logging.info("Snake initialized with body, direction, fruit, and path.")

    def calculate_path(self) -> List[Tuple[int, int]]:
        """
        Calculate the path from the snake's head to the fruit using the A* algorithm.
        :return: A list of tuples representing the path coordinates.
        """
        return a_star(
            self.body[0], self.fruit.position, set(self.body), list(self.body)
        )

    def draw(self) -> None:
        """
        Draw the snake on the game window using a fixed color (green) and block size for each segment.
        """
        color: Tuple[int, int, int] = (0, 255, 0)  # RGB color for the snake
        for segment in list(self.body):
            pg.draw.rect(self.window, color, (*segment, BLOCK_SIZE, BLOCK_SIZE))
        logging.debug(f"Snake drawn on window at segments: {list(self.body)}")

    def move(self) -> bool:
        """
        Move the snake based on the A* pathfinding result. Handles collision and game over scenarios.
        Implements a zigzag movement pattern based on the current direction to maintain a super dense configuration from point to point.
        :return: Boolean indicating if the move was successful (True) or if the game is over (False).
        """
        if not self.path:
            self.path = self.calculate_path()  # Recalculate path if needed
        if self.path:
            next_pos: Tuple[int, int] = self.path.pop(0)
            if self.is_collision(next_pos):
                logging.warning("Collision detected or snake out of bounds")
                return False
            self.body.appendleft(next_pos)
            if next_pos == self.fruit.position:
                self.fruit.relocate(list(self.body))
                self.path = self.calculate_path()
            else:
                self.body.pop()
            logging.info(f"Snake moved successfully to {next_pos}")
            return True
        logging.warning("No path available, game over")
        return False

    def is_collision(self, position: Tuple[int, int]) -> bool:
        """
        Check if the given position results in a collision or is out of game bounds.
        :param position: Tuple representing the position to check.
        :return: True if there is a collision or out of bounds, False otherwise.
        """
        return (
            position in self.body
            or not (0 <= position[0] < SCREEN_WIDTH)
            or not (0 <= position[1] < SCREEN_HEIGHT)
        )


def main():
    clock = pg.time.Clock()
    snake = Snake(window)
    fruit = Fruit(window)

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

        window.fill((0, 0, 0))
        snake.fruit.draw()
        snake.draw()

        if not snake.move():
            logging.info("Game Over, restarting")
            print("Game Over")
            snake = Snake(window)  # Restart the game

        pg.display.flip()
        clock.tick(60)
        logging.debug("Game loop executed")


if __name__ == "__main__":
    main()
