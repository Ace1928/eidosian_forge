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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
import multiprocessing
from queue import PriorityQueue
from typing import Set, List, Tuple, Dict, Deque
from collections import deque
import math

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
from collections import deque
from heapq import heappush, heappop


def compute_square_difference(x: int, y: int) -> int:
    """
    Compute the square of the difference between two integers.
    This function is designed to be used in parallel processing environments to maximize efficiency.
    """
    return (x - y) ** 2


def euclidean_distance(node1: Tuple[int, int], node2: Tuple[int, int]) -> float:
    """
    Calculate the Euclidean distance between two points using an optimized parallel approach.
    This function maximizes the utilization of available CPU resources by employing multiprocessing.
    """
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        future_x = executor.submit(compute_square_difference, node1[0], node2[0])
        future_y = executor.submit(compute_square_difference, node1[1], node2[1])
        result_x = future_x.result()
        result_y = future_y.result()
    distance = np.sqrt(result_x + result_y)
    return distance


def heuristic(
    node1: Tuple[int, int],
    node2: Tuple[int, int],
    snake_body: List[Tuple[int, int]],
    fruit_position: Tuple[int, int],
) -> float:
    """
    An advanced heuristic function that combines Euclidean distance with game-specific adjustments.
    This heuristic is designed to be both efficient and effective for the snake game's pathfinding needs.
    """
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2) as executor:
        base_distance_future = executor.submit(euclidean_distance, node1, node2)
        body_proximity_futures = [
            executor.submit(math.exp, -euclidean_distance(node1, body_part))
            for body_part in snake_body
        ]
        center_of_mass_future = executor.submit(
            lambda: tuple(np.mean(np.array(snake_body), axis=0))
        )
        base_distance = base_distance_future.result()
        body_proximity_penalty = sum(f.result() for f in body_proximity_futures)
        center_of_mass = center_of_mass_future.result()
        density_bonus_future = executor.submit(
            lambda: -np.linalg.norm(np.array(node1) - np.array(center_of_mass))
        )
        fruit_priority_future = executor.submit(
            lambda: (
                -euclidean_distance(node1, fruit_position)
                if np.array_equal(node1, center_of_mass)
                else 0
            )
        )
        if len(snake_body) > 1:
            direction_vector = np.subtract(snake_body[0], snake_body[1])
            zigzag_penalty_future = executor.submit(
                lambda: -abs(
                    np.cross(direction_vector, np.subtract(node1, snake_body[0]))
                )
            )
            zigzag_penalty = zigzag_penalty_future.result()
        else:
            zigzag_penalty = 0
        density_bonus = density_bonus_future.result()
        fruit_priority = fruit_priority_future.result()
    return (
        base_distance
        + 10 * body_proximity_penalty
        + 5 * density_bonus
        + 20 * fruit_priority
        + 15 * zigzag_penalty
    )


def is_within_boundaries(
    node: Tuple[int, int], screen_width: int, screen_height: int
) -> bool:
    """
    Check if a node is within the game boundaries.
    """
    x, y = node
    return 0 <= x < screen_width and 0 <= y < screen_height


def get_neighbors(
    node: Tuple[int, int],
    block_size: int = BLOCK_SIZE,
    screen_width: int = SCREEN_WIDTH,
    screen_height: int = SCREEN_HEIGHT,
) -> List[Tuple[int, int]]:
    """
    Return the neighbors of the given node within game boundaries.
    """
    directions = [(block_size, 0), (-block_size, 0), (0, block_size), (0, -block_size)]
    with ThreadPoolExecutor(max_workers=len(directions)) as executor:
        future_to_direction = {
            executor.submit(
                is_within_boundaries,
                (node[0] + dx, node[1] + dy),
                screen_width,
                screen_height,
            ): (dx, dy)
            for dx, dy in directions
        }
        neighbors = [
            (node[0] + dx, node[1] + dy)
            for future, (dx, dy) in future_to_direction.items()
            if future.result()
        ]
    return neighbors


def a_star(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    snake_body: List[Tuple[int, int]],
    fruit_position: Tuple[int, int],
    path_so_far: Optional[List[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """
    Perform A* pathfinding avoiding given obstacles and considering the snake's body with strategic prioritization.
    Tracks the path taken so far to enhance decision-making and debugging.

    :param start: Starting node as a tuple of (x, y).
    :param goal: Goal node as a tuple of (x, y).
    :param obstacles: Set of tuples representing the positions of obstacles.
    :param snake_body: List of tuples representing the snake's body positions.
    :param fruit_position: Tuple representing the position of the fruit.
    :param path_so_far: List of nodes representing the path taken so far.
    :return: List of nodes representing the optimal path from start to goal.
    """
    if path_so_far is None:
        path_so_far = []

    open_set = PriorityQueue()
    open_set.put((heuristic(start, goal, snake_body, fruit_position), start))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal, snake_body, fruit_position)}
    max_workers = multiprocessing.cpu_count() * 2

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        while not open_set.empty():
            _, current = open_set.get()
            path_so_far.append(current)  # Append current node to path_so_far
            futures.append(
                executor.submit(
                    process_node,
                    current,
                    f_score[current],
                    open_set,
                    came_from,
                    g_score,
                    f_score,
                    snake_body,
                    fruit_position,
                    goal,
                    path_so_far,
                )
            )
        optimal_paths = [
            future.result() for future in as_completed(futures) if future.result()
        ]
        if optimal_paths:
            best_path = min(optimal_paths, key=lambda path: len(path))
            return optimize_path_for_density(best_path, snake_body)
    return []


def process_node(
    current: Tuple[int, int],
    current_f_score: float,
    open_set: PriorityQueue,
    came_from: Dict[Tuple[int, int], Tuple[int, int]],
    g_score: Dict[Tuple[int, int], float],
    f_score: Dict[Tuple[int, int], float],
    snake_body: List[Tuple[int, int]],
    fruit_position: Tuple[int, int],
    goal: Tuple[int, int],
    path_so_far: List[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    """
    Process a node in the A* pathfinding algorithm, considering the current node's score and updating paths and scores accordingly.
    Now also tracks the path taken so far for enhanced decision-making.

    :param current: The current node being processed.
    :param current_f_score: The F score of the current node.
    :param open_set: The priority queue representing the open set in A*.
    :param came_from: A dictionary mapping each node to its predecessor in the path.
    :param g_score: A dictionary mapping each node to its G score.
    :param f_score: A dictionary mapping each node to its F score.
    :param snake_body: The list of tuples representing the snake's body coordinates.
    :param fruit_position: The position of the fruit as a tuple.
    :param goal: The goal node as a tuple.
    :param path_so_far: List of nodes representing the path taken so far.
    :return: A list of nodes representing the path from the start to the current node if the goal is reached, otherwise None.
    """
    if current == goal:
        return reconstruct_path(came_from, current)

    neighbors = get_neighbors(current)
    for neighbor in neighbors:
        if neighbor in snake_body:
            continue  # Skip the neighbor if it is part of the snake's body

        tentative_g_score = g_score[current] + 1  # Assuming uniform cost for simplicity
        if tentative_g_score < g_score.get(neighbor, float("inf")):
            # This path to neighbor is better than any previous one. Record it!
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = tentative_g_score + heuristic(
                neighbor, goal, snake_body, fruit_position
            )
            open_set.put((f_score[neighbor], neighbor))

    return None  # If the goal is not reached, return None


def reconstruct_path(
    came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Reconstructs the path from the start node to the goal node using the came_from map.
    """
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


def process_path_segment(
    next_position: Tuple[int, int],
    center_of_mass: Tuple[float, float],
    snake_body: List[Tuple[int, int]],
    current_position: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    Process a segment of the path for optimal placement in relation to the center of mass and other game logic.

    :param next_position: The next position to process, represented as a tuple of integers (x, y).
    :param center_of_mass: The current center of mass of the snake, represented as a tuple of floats (x, y).
    :param snake_body: The current state of the snake's body, represented as a list of tuples of integers [(x1, y1), (x2, y2), ...].
    :param current_position: The current position of the snake, represented as a tuple of integers (x, y).
    :return: A list of positions representing the processed path segment, each position is a tuple of integers (x, y).
    """
    # Determine if the next position optimizes the density of the snake's body
    if is_position_optimal_for_density(next_position, center_of_mass, snake_body):
        # If optimal, return a path segment leading directly to the next position
        return [current_position, next_position]
    else:
        # If not optimal, find an alternative path that maintains density
        alternative_path = find_alternative_path(
            current_position, next_position, snake_body, center_of_mass
        )
        if alternative_path:
            return alternative_path
        else:
            # If no alternative path is found, default to moving directly to the next position
            return [current_position, next_position]


def optimize_path_for_density(
    path: List[Tuple[int, int]], snake_body: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Optimizes the path to ensure maximum density of the snake's body.
    """
    optimized_path = []
    center_of_mass = calculate_center_of_mass(snake_body)
    current_position = path[0]
    path_queue = deque(path)
    max_workers = multiprocessing.cpu_count() * 2
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        while path_queue:
            next_position = path_queue.popleft()
            futures.append(
                executor.submit(
                    process_path_segment,
                    next_position,
                    center_of_mass,
                    snake_body,
                    current_position,
                )
            )
        for future in as_completed(futures):
            result = future.result()
            if result:
                optimized_path.extend(result)
                current_position = result[-1]
    return optimized_path


def calculate_center_of_mass(snake_body: List[Tuple[int, int]]) -> Tuple[float, float]:
    """
    Calculate the geometric center of the snake's body.
    """
    snake_body_np = np.array(snake_body)
    center_x, center_y = np.mean(snake_body_np, axis=0)
    return (float(center_x), float(center_y))


def is_position_optimal_for_density(
    position: Tuple[int, int],
    center_of_mass: Tuple[float, float],
    snake_body: List[Tuple[int, int]],
) -> bool:
    """
    Determine if a position contributes to the optimal density of the snake's body.
    """
    position_np = np.array(position)
    center_of_mass_np = np.array(center_of_mass)
    snake_body_np = np.array(snake_body)
    distance_to_center = np.linalg.norm(position_np - center_of_mass_np)
    return (
        distance_to_center < np.linalg.norm(snake_body_np[-1] - center_of_mass_np)
        and tuple(position) not in snake_body
    )


def find_alternative_path(
    current: Tuple[int, int],
    target: Tuple[int, int],
    snake_body: List[Tuple[int, int]],
    center_of_mass: Tuple[float, float],
) -> List[Tuple[int, int]]:
    """
    Find an alternative path that maintains density when the direct path is not optimal.
    """
    max_workers = multiprocessing.cpu_count() * 2
    path_results = deque()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        open_set = []
        heappush(open_set, (0, current))
        came_from = {}
        g_score = {current: 0}
        f_score = {current: np.linalg.norm(np.array(current) - np.array(target))}
        while open_set:
            _, current = heappop(open_set)
            if current == target:
                path_results.appendleft(reconstruct_path(came_from, current))
                continue
            neighbors = get_neighbors(current)
            for neighbor in neighbors:
                if neighbor in snake_body:
                    continue
                futures.append(
                    executor.submit(
                        process_neighbor,
                        neighbor,
                        current,
                        target,
                        center_of_mass,
                        g_score,
                        came_from,
                    )
                )
        for future in as_completed(futures):
            neighbor, tentative_g_score, came_from_update = future.result()
            if tentative_g_score < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = came_from_update
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + np.linalg.norm(
                    np.array(neighbor) - np.array(center_of_mass)
                )
                heappush(open_set, (f_score[neighbor], neighbor))
    return list(path_results)


def process_neighbor(neighbor, current, target, center_of_mass, g_score, came_from):
    """
    Process each neighbor in the A* algorithm, intended to be run in a separate thread.
    """
    tentative_g_score = g_score[current] + 1
    return neighbor, tentative_g_score, current


def process_node(
    current: Tuple[int, int],
    current_f_score: float,
    open_set: PriorityQueue,
    came_from: Dict[Tuple[int, int], Tuple[int, int]],
    g_score: Dict[Tuple[int, int], float],
    f_score: Dict[Tuple[int, int], float],
    snake_body: List[Tuple[int, int]],
    fruit_position: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    Process a node in the A* pathfinding algorithm, considering the current node's score and updating paths and scores accordingly.

    :param current: The current node being processed.
    :param current_f_score: The F score of the current node.
    :param open_set: The priority queue representing the open set in A*.
    :param came_from: A dictionary mapping each node to its predecessor in the path.
    :param g_score: A dictionary mapping each node to its G score.
    :param f_score: A dictionary mapping each node to its F score.
    :param snake_body: The list of tuples representing the snake's body coordinates.
    :param fruit_position: The position of the fruit as a tuple.
    :param goal: The goal node as a tuple.
    :return: A list of nodes representing the path from the start to the current node if the goal is reached, otherwise None.
    """
    if current == goal:
        return reconstruct_path(came_from, current)

    neighbors = get_neighbors(current)
    for neighbor in neighbors:
        if neighbor in snake_body:
            continue  # Skip the neighbor if it is part of the snake's body

        tentative_g_score = g_score[current] + 1  # Assuming uniform cost for simplicity

        if tentative_g_score < g_score.get(neighbor, float("inf")):
            # This path to neighbor is better than any previous one. Record it!
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = tentative_g_score + heuristic(
                neighbor, goal, snake_body, fruit_position
            )
            open_set.put((f_score[neighbor], neighbor))

    return None  # If the goal is not reached, return None


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
        """
        self.window = window
        self.body: Deque[Tuple[int, int]] = deque([(160, 160), (140, 160), (120, 160)])
        self.direction: Tuple[int, int] = (BLOCK_SIZE, 0)  # Moving right initially
        self.last_direction: Tuple[int, int] = (
            self.direction
        )  # Store the last direction
        self.fruit: Fruit = Fruit(window)
        self.path: List[Tuple[int, int]] = self.calculate_path()

    def calculate_path(self) -> List[Tuple[int, int]]:
        """
        Calculate the path from the snake's head to the fruit using the A* algorithm.
        This method ensures that the snake will attempt to find a path to the fruit. If no path is found,
        it will continue moving in the last valid direction to avoid stalling.
        """
        # Attempt to find a path using the A* algorithm
        path = a_star(
            start=self.body[0],  # Starting position is the head of the snake
            goal=self.fruit.position,  # Goal is the position of the fruit
            obstacles=set(self.body),  # The snake's body serves as obstacles
            path_so_far=list(self.body),  # Current path is the snake's body
            target=self.fruit.position,  # Target is again the position of the fruit
        )

        # If the A* algorithm fails to find a path, continue in the last known direction
        if not path:
            logging.debug("No path found using A*, continuing in last direction.")
            head_x, head_y = self.body[0]
            next_pos = (
                head_x
                + self.last_direction[
                    0
                ],  # Continue moving in the last horizontal direction
                head_y
                + self.last_direction[
                    1
                ],  # Continue moving in the last vertical direction
            )
            path = [
                next_pos
            ]  # The path is now just the next position in the last known direction

        # Log the calculated or continued path for debugging purposes
        logging.info(f"Path calculated or continued: {path}")
        return path

    def move(self) -> bool:
        """
        Move the snake based on the A* pathfinding result. Handles collision and game over scenarios.
        """
        if not self.path:
            self.path = self.calculate_path()  # Recalculate path if needed
        if self.path:
            next_pos: Tuple[int, int] = self.path.pop(0)
        else:
            # Continue in the last direction if no path is found
            head_x, head_y = self.body[0]
            next_pos = (
                head_x + self.last_direction[0],
                head_y + self.last_direction[1],
            )

        if self.is_collision(next_pos):
            logging.warning("Collision detected or snake out of bounds")
            return False

        self.body.appendleft(next_pos)
        self.last_direction = (
            next_pos[0] - self.body[1][0],
            next_pos[1] - self.body[1][1],
        )  # Update last direction

        if next_pos == self.fruit.position:
            self.fruit.relocate(list(self.body))
            self.path = self.calculate_path()  # Recalculate path after eating the fruit
        else:
            self.body.pop()

        logging.info(f"Snake moved successfully to {next_pos}")
        return True

    def draw(self) -> None:
        """
        Draw the snake on the game window using a fixed color (green) and block size for each segment.
        """
        color: Tuple[int, int, int] = (0, 255, 0)  # RGB color for the snake
        for segment in list(self.body):
            pg.draw.rect(self.window, color, (*segment, BLOCK_SIZE, BLOCK_SIZE))
        logging.debug(f"Snake drawn on window at segments: {list(self.body)}")

    def is_collision(self, position: Tuple[int, int]) -> bool:
        """
        Check if the given position results in a collision or is out of game bounds.
        Specifically checks if the head collides with the body or the boundaries.
        """
        # Check if the head collides with any part of the body except the last segment (tail)
        body_without_tail = list(self.body)[:-1]
        return (
            position in body_without_tail
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
